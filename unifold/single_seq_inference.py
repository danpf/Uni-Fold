#!/usr/bin/env python

import argparse
import pathlib
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import sys
import logging
from dataclasses import dataclass
import os
import pickle
import gzip
import json
from unifold.config import model_config
from unifold.modules.alphafold import AlphaFold
from unifold.dataset import load_and_process, UnifoldDataset
import time


import torch
import numpy as np

from unifold.msa import templates, pipeline, parsers
from unifold.msa.utils import divide_multi_chains

from unifold.data.utils import compress_features
from unifold.data import residue_constants, protein

from unifold.data.protein import PDB_CHAIN_IDS

from unicore.utils import (
    tensor_tree_map,
)



log = logging.getLogger(__name__)

@dataclass
class ToFoldInput:
    _seq: str
    id_: str

    @property
    def seqs(self) -> List[str]:
        """The seqs property."""
        return self._seqs

    @seqs.setter
    def seqs(self, value: List[str]):
        self._seqs = value

    def __post_init__(self):
        self.seqs = self._seq.split(":")

@dataclass
class FoldSettings:
    num_ensembles: int
    max_recycling_iters: int
    manual_seed: int
    times: int


def parseargs(args_in: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequences", nargs="+", help="Sequences to fold, use : in  the sequence to specify more than one chain")
    parser.add_argument("--sequence-ids", nargs="+", help="Names of sequences to fold default is just the index")
    parser.add_argument("--output-dir", default="output-fold", help="Names of sequences to fold default is just the index")
    parser.add_argument("--a3m-files", nargs="+", help="the a3m files to use (will use the same for every sequence)")
    parser.add_argument("--max-recycling-iters", type=int, default=3, help="the a3m files to use (will use the same for every sequence)")
    parser.add_argument("--num-ensembles", type=int, default=2, help="the a3m files to use (will use the same for every sequence)")
    parser.add_argument("--manual-seed", type=int, default=42, help="the a3m files to use (will use the same for every sequence)")
    parser.add_argument("--jumpstart-pdb-file", help="Template PDB file to use (numbering from 0 must match)")
    parser.add_argument("--times", type=int, default=3, help="the a3m files to use (will use the same for every sequence)")
    args = parser.parse_args(args_in)
    if args.sequence_ids:
        assert len(args.sequence_ids) == len(args.sequences)
    return args


def get_null_template(
    query_sequence: Union[List[str], str], num_temp: int = 1
) -> Dict[str, Any]:
    ln = (
        len(query_sequence)
        if isinstance(query_sequence, str)
        else sum(len(s) for s in query_sequence)
    )
    output_templates_sequence = "A" * ln

    templates_all_atom_positions = np.zeros(
        (ln, templates.residue_constants.atom_type_num, 3)
    )
    templates_all_atom_masks = np.zeros((ln, templates.residue_constants.atom_type_num))
    templates_aatype = templates.residue_constants.sequence_to_onehot(
        output_templates_sequence, templates.residue_constants.HHBLITS_AA_TO_ID
    )
    template_features = {
        "template_all_atom_positions": np.tile(
            templates_all_atom_positions[None], [num_temp, 1, 1, 1]
        ),
        "template_all_atom_masks": np.tile(
            templates_all_atom_masks[None], [num_temp, 1, 1]
        ),
        "template_sequence": [f"none".encode()] * num_temp,
        "template_aatype": np.tile(np.array(templates_aatype)[None], [num_temp, 1, 1]),
        "template_domain_names": [f"none".encode()] * num_temp,
        "template_sum_probs": np.zeros([num_temp], dtype=np.float32),
    }
    return template_features


def get_single_seq_msa_and_templates(
    unique_seqs: Sequence[str],
) -> Tuple[
    List[str], List[str], List[Dict[str, Any]] # type wrong
]:
    
    template_features = [get_null_template(x) for x in unique_seqs]
    a3m_text_per_template = []
    num = 101
    for i, seq in enumerate(unique_seqs):
        a3m_text_per_template.append(">" + str(num + i) + "\n" + seq)

    # tlen = sum(len(x) for x in unique_seqs)
    # start = 0
    # paired_a3m_lines = []
    # for i, seq in enumerate(unique_seqs):
    #     out = list("-"*tlen)
    #     out[start:start+len(seq)] = list(seq)
    #     start += len(seq)
    #     a3m_text_per_template.append(">" + str(num + i) + "\n" + "".join(out))

    return (
        a3m_text_per_template,
        a3m_text_per_template,
        template_features,
    )

def get_a3m_lines(files: Sequence[str], ignore_first: bool = True) -> List[str]:
    out_lines = []
    for fn in files:
        with open(fn) as fh:
            txt_lines = [x.strip() for x in fh.read().split("\n") if x.strip() and not x.startswith("#")]
        if ignore_first:
            txt_lines = txt_lines[2:]
        out_lines += txt_lines
    return out_lines


def main(seqs_to_fold: Sequence[ToFoldInput], output_dir_master: str, a3m_lines: Sequence[str], jumpstart_pdb_file: str, foldsettings: FoldSettings):
    if a3m_lines:
        assert False, "Broken"
    jumpstart_protein = None
    if jumpstart_pdb_file:
        jumpstart_protein = protein.from_pdb_string(open(jumpstart_pdb_file).read())

    is_multimer = False
    for seq_to_fold in seqs_to_fold:

        output_dir = os.path.join(output_dir_master, str(seq_to_fold.id_))
        pathlib.Path(output_dir).mkdir(exist_ok=True, parents=True)
        if len(seq_to_fold.seqs) > 1:
            is_multimer = True
        unique_seqs = []
        _ = [unique_seqs.append(x) for x in seq_to_fold.seqs if x not in unique_seqs]

        if is_multimer:
            divide_multi_chains(seq_to_fold.id_, output_dir_master, seq_to_fold.seqs, [f">{seq_to_fold.id_}_{i}" for i in range(len(seq_to_fold.seqs))])


        unpaired_msa,          paired_msa,          template_results = get_single_seq_msa_and_templates(unique_seqs)
        
        for idx, seq in enumerate(unique_seqs):
            chain_id = PDB_CHAIN_IDS[idx]
            sequence_features = pipeline.make_sequence_features(
                      sequence=seq, description=f'>{seq_to_fold.id_} seq {chain_id}', num_res=len(seq)
                  )
            monomer_msa = parsers.parse_a3m(unpaired_msa[idx])
            msa_features = pipeline.make_msa_features([monomer_msa])
            template_features = template_results[idx]
            feature_dict = {**sequence_features, **msa_features, **template_features}
            feature_dict = compress_features(feature_dict)
            features_output_path = os.path.join(
                    output_dir, "{}.feature.pkl.gz".format(chain_id)
                )
            pickle.dump(
                feature_dict, 
                gzip.GzipFile(features_output_path, "wb"), 
                protocol=4
                )
            if len(seq_to_fold.seqs) > 1:
                multimer_msa = parsers.parse_a3m(paired_msa[idx])
                pair_features = pipeline.make_msa_features([multimer_msa])
                pair_feature_dict = compress_features(pair_features)
                uniprot_output_path = os.path.join(
                    output_dir, "{}.uniprot.pkl.gz".format(chain_id)
                )
                pickle.dump(
                    pair_feature_dict,
                    gzip.GzipFile(uniprot_output_path, "wb"),
                    protocol=4,
                )
    if is_multimer:
        model_name = "multimer_ft"
        param_path= "/home/mable/git/Uni-Fold/params/multimer.unifold.pt"
    else:
        model_name = "model_2_ft"
        param_path = "/home/mable/git/Uni-Fold/params/monomer.unifold.pt"

    max_recycling_iters = foldsettings.max_recycling_iters #@param {type:"integer"}
    num_ensembles = foldsettings.num_ensembles #@param {type:"integer"}
    manual_seed = foldsettings.manual_seed #@param {type:"integer"}
    times = foldsettings.times #@param {type:"integer"}

    config = model_config(model_name)
    config.data.common.max_recycling_iters = max_recycling_iters
    config.globals.max_recycling_iters = max_recycling_iters
    config.data.predict.num_ensembles = num_ensembles
# faster prediction with large chunk
    config.globals.chunk_size = 128
    model = AlphaFold(config)
    print("start to load params {}".format(param_path))
    state_dict = torch.load(param_path)["ema"]["params"]
    state_dict = {".".join(k.split(".")[1:]): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model = model.to("cuda:0")
    model.eval()
    model.inference_mode()


    def _dump_every_recycle(base_fn: str):
        def dump_every_recycle(batch, out, iter_: int, model: AlphaFold):
            t = time.perf_counter()
            
            out.update(model.aux_heads(out))
            def to_float(x):
                if x.dtype == torch.bfloat16 or x.dtype == torch.half:
                    return x.float()
                else:
                    return x
            # Toss out the recycling dimensions --- we don't need them anymore
            batch = tensor_tree_map(lambda t: t[-1, 0, ...], batch)
            batch = tensor_tree_map(to_float, batch)
            out = tensor_tree_map(lambda t: t[0, ...], out)
            out = tensor_tree_map(to_float, out)
            batch = tensor_tree_map(lambda x: np.array(x.cpu()), batch)
            out = tensor_tree_map(lambda x: np.array(x.cpu()), out)
            plddt = out["plddt"]
            plddt_b_factors = np.repeat(
                plddt[..., None], residue_constants.atom_type_num, axis=-1
            )
            # TODO: , may need to reorder chains, based on entity_ids
            cur_protein = protein.from_prediction(
                features=batch, result=out, b_factors=plddt_b_factors
            )
            save_name = f"{base_fn}_recy-{iter_:02}.pdb"
            print(f"time to write pdb time: {time.perf_counter() - t:.1f} sec -- Writing... {save_name}")
            with open(save_name, "w") as f:
                f.write(protein.to_pdb(cur_protein))
        return dump_every_recycle

    # data path is based on target_name
    cur_param_path_postfix = os.path.split(param_path)[-1]
    for seq_to_fold in seqs_to_fold:
        output_dir = os.path.join(output_dir_master, str(seq_to_fold.id_))

        plddts = {}
        ptms = {}
        best_protein = None
        best_score = 0
        best_plddt = None
        best_pae = None

        print("start to predict {}".format(seq_to_fold.id_))
        for seed in range(times):
            cur_seed = hash((manual_seed, seed)) % 100000
            batch = load_feature_for_one_target(
                config,
                output_dir,
                cur_seed,
                is_multimer=is_multimer,
                use_uniprot=is_multimer,
            )
            seq_len = batch["aatype"].shape[-1]
            if jumpstart_protein:
                batch["x_prev"] = jumpstart_protein.atom_positions
            model.globals.chunk_size = automatic_chunk_size(seq_len)

            cur_save_name = (
                f"{cur_param_path_postfix}_{cur_seed}"
            )

            with torch.no_grad():
                batch = {
                    k: torch.as_tensor(v, device="cuda:0")
                    for k, v in batch.items()
                }
                shapes = {k: v.shape for k, v in batch.items()}
                print(shapes)
                t = time.perf_counter()
                out = model(batch, _dump_every_recycle(os.path.join(output_dir, cur_save_name)))
                print(f"Inference time: {time.perf_counter() - t}")

            def to_float(x):
                if x.dtype == torch.bfloat16 or x.dtype == torch.half:
                    return x.float()
                else:
                    return x

            # Toss out the recycling dimensions --- we don't need them anymore
            batch = tensor_tree_map(lambda t: t[-1, 0, ...], batch)
            batch = tensor_tree_map(to_float, batch)
            out = tensor_tree_map(lambda t: t[0, ...], out)
            out = tensor_tree_map(to_float, out)
            batch = tensor_tree_map(lambda x: np.array(x.cpu()), batch)
            out = tensor_tree_map(lambda x: np.array(x.cpu()), out)

            plddt = out["plddt"]
            mean_plddt = np.mean(plddt)
            plddt_b_factors = np.repeat(
                plddt[..., None], residue_constants.atom_type_num, axis=-1
            )
            # TODO: , may need to reorder chains, based on entity_ids
            cur_protein = protein.from_prediction(
                features=batch, result=out, b_factors=plddt_b_factors
            )
            plddts[cur_save_name] = str(mean_plddt)
            if is_multimer:
                ptms[cur_save_name] = str(np.mean(out["iptm+ptm"]))
            with open(os.path.join(output_dir, cur_save_name + '.pdb'), "w") as f:
                f.write(protein.to_pdb(cur_protein))

            if is_multimer:
                mean_ptm = np.mean(out["iptm+ptm"])
                if mean_ptm>best_score:
                    best_protein = cur_protein
                    best_pae = out["predicted_aligned_error"]
                    best_plddt = out["plddt"]
                    best_score = mean_ptm
            else:
                if mean_plddt>best_score:
                    best_protein = cur_protein
                    best_plddt = out["plddt"]
                    best_score = mean_plddt

        print("plddts", plddts)
        score_name = f"{model_name}_{cur_param_path_postfix}"
        plddt_fname = score_name + "_plddt.json"
        json.dump(plddts, open(os.path.join(output_dir, plddt_fname), "w"), indent=4)
        if ptms:
            print("ptms", ptms)
            ptm_fname = score_name + "_ptm.json"
            json.dump(ptms, open(os.path.join(output_dir, ptm_fname), "w"), indent=4)

def automatic_chunk_size(seq_len):
    if seq_len < 512:
        chunk_size = 256
    elif seq_len < 1024:
        chunk_size = 128
    elif seq_len < 2048:
        chunk_size = 32
    elif seq_len < 3072:
        chunk_size = 16
    else:
        chunk_size = 1
    return chunk_size

def load_feature_for_one_target(
    config, data_folder, seed=0, is_multimer=False, use_uniprot=False
):
    if not is_multimer:
        uniprot_msa_dir = None
        sequence_ids = ["A"]
        if use_uniprot:
            uniprot_msa_dir = data_folder

    else:
        uniprot_msa_dir = data_folder
        sequence_ids = open(os.path.join(data_folder, "chains.txt")).readline().split()
    batch, _ = load_and_process(
        config=config.data,
        mode="predict",
        seed=seed,
        batch_idx=None,
        data_idx=0,
        is_distillation=False,
        sequence_ids=sequence_ids,
        monomer_feature_dir=data_folder,
        uniprot_msa_dir=uniprot_msa_dir,
    )
    batch = UnifoldDataset.collater([batch])
    return batch


def commandline_main(_args: List[str]) -> None:
    args = parseargs(_args)
    foldsettings = FoldSettings(args.num_ensembles, args.max_recycling_iters, args.manual_seed, args.times)

    if args.sequence_ids:
        seqs_to_fold = [ToFoldInput(x, y) for x, y in zip(args.sequences, args.sequence_ids)]
    else:
        seqs_to_fold = [ToFoldInput(x, f"{i:04}") for i, x in enumerate(args.sequences)]
    # a3m_lines = get_a3m_lines(args.a3m_lines)
    # a3m_lines = []
    main(seqs_to_fold, args.output_dir, a3m_lines=[], jumpstart_pdb_file=args.jumpstart_pdb_file, foldsettings=foldsettings)


if __name__ == "__main__":
    commandline_main(sys.argv[1:])

