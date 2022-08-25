import gzip
import logging
from typing import Any, Dict, List, Union
import numpy as np
import os
import argparse
from dataclasses import dataclass, field
import json
import pickle
import time
from pathlib import Path
import tempfile

from simple_parsing import ArgumentParser
from simple_parsing.helpers.fields import subparsers


# import torch
from unifold.config import model_config
from unifold.modules.alphafold import AlphaFold
from unifold.data import residue_constants, protein
from unifold.dataset import load_and_process, UnifoldDataset

from unifold.data.utils import compress_features, divide_multi_chains
from unifold.msa import parsers, pipeline, templates
from unifold.msa.tools import hmmsearch
from .homo_search import generate_pkl_features

# from unicore.utils import (
#     tensor_tree_map,
# )
@dataclass
class StandardArgs:
    sample_templates: bool = False
    use_uniprot: bool = False


@dataclass
class NoMSAArgs:
    sequences: List[str]


@dataclass
class GlobalInferenceArgs:
    subcommand: Union[StandardArgs, NoMSAArgs] = subparsers(
        {"standard": StandardArgs, "nomsa": NoMSAArgs}, default=StandardArgs()
    )

    param_path: str = ""

    model_device: str = "cuda:0"
    """Name of the device on which to run the model. Any valid torch device name is accepted (e.g. "cpu", "cuda:0")"""

    model_names: List[str] = field(default_factory=lambda: ["model_2"])
    """Names of models to use"""

    data_random_seed: int = 42

    data_dirs: List[str] = field(default_factory=lambda: [])
    target_names: List[str] = field(default_factory=lambda: [])
    output_dir: str = ""

    times: int = 3
    """Number of times to fold with each model"""

    max_recycling_iters: int = 3
    """Number of times to recycle"""

    num_ensembles: int = 2

    bf16: bool = False
    save_raw_output: bool = False

    def __post_init__(self):
        assert self.param_path
        assert self.output_dir
        assert len(self.data_dirs) == len(self.target_names)


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


def create_fake_af_folder_structure_and_get_pipeline(base_dir: str):
    bdp = Path(base_dir)
    for x in ["jackhmmer_binary_path", "hhblits_binary_path", "hmmsearch_binary_path", "hmmbuild_binary_path", "kalign_binary_path"]:
        with open(bdp / x, "w") as fh:
            fh.write("")
    for x in ["uniref90_database_path",
        "mgnify_database_path",
        "bfd_database_path",
        "uniclust30_database_path",
        "small_bfd_database_path",
        "uniprot_database_path",
        "hmm_database_path",
        "template_mmcif_dir",
        "obsolete_pdbs_path"]:
        (bdp/x).mkdir(exist_ok=True,parents=True)

    template_searcher = hmmsearch.Hmmsearch(
        binary_path="hmmsearch_binary_path",
        hmmbuild_binary_path="hmmbuild_binary_path",
        database_path="hmm_database_path",
    )
    template_featurizer = templates.HmmsearchHitFeaturizer(
        mmcif_dir="template_mmcif_dir",
        max_template_date="2090-01-01",
        max_hits=9999,
        kalign_binary_path="kalign_binary_path",
        release_dates_path=None,
        obsolete_pdbs_path="obsolete_pdbs_path",
    )

    data_pipeline = pipeline.DataPipeline(
        jackhmmer_binary_path="jackhmmer_binary_path",
        hhblits_binary_path="hhblits_binary_path",
        uniref90_database_path="uniref90_database_path",
        mgnify_database_path="mgnify_database_path",
        bfd_database_path="bfd_database_path",
        uniclust30_database_path="uniclust30_database_path",
        small_bfd_database_path="small_bfd_database_path",
        uniprot_database_path="uniprot_database_path",
        template_searcher=template_searcher,
        template_featurizer=template_featurizer,
        use_small_bfd=False,
        use_precomputed_msas=True,
    )
    return data_pipeline




def create_nomsa_features(config, sequence: str, seed: int=0, is_multimer: bool = False):
    if not is_multimer:
        uniprot_msa_dir = None
        sequence_ids = ["A"]
    else:
        uniprot_msa_dir = ""
        alphabet = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
        seqs = sequence.split(":")
        sequence_ids = alphabet[:len(seqs)]
    with tempfile.TemporaryDirectory() as tmpdir:
        data_pipeline = create_fake_af_folder_structure(tmpdir)
        fasta_path = "seqin.fasta"
        with open(fasta_path, "w") as fh:
            fh.write(f">seqin\n{sequence}")
        generate_pkl_features(fasta_path, "seqin", tmpdir, data_pipeline, False)

    # templates_result = self.template_featurizer.get_templates(
    #     query_sequence=input_sequence, hits=pdb_template_hits
    # )
    #
    # sequence_features = make_sequence_features(
    #     sequence=input_sequence, description=input_description, num_res=num_res
    # )
    #
    # msa_features = make_msa_features((uniref90_msa, bfd_msa, mgnify_msa))
    #
    # logging.info("Uniref90 MSA size: %d sequences.", len(uniref90_msa))
    # logging.info("BFD MSA size: %d sequences.", len(bfd_msa))
    # logging.info("MGnify MSA size: %d sequences.", len(mgnify_msa))
    # logging.info(
    #     "Final (deduplicated) MSA size: %d sequences.",
    #     msa_features["num_alignments"][0],
    # )
    # logging.info(
    #     "Total number of templates (NB: this can include bad "
    #     "templates and is later filtered to top 4): %d.",
    #     templates_result.features["template_domain_names"].shape[0],
    # )
    #
    # return {**sequence_features, **msa_features, **templates_result.features}


    batch, _ = load_and_process(
        config=config.data,
        mode="predict",
        seed=seed,
        batch_idx=None,
        data_idx=0,
        is_distillation=False,
        sequence_ids=sequence_ids,
        monomer_feature_dir="",
        uniprot_msa_dir=uniprot_msa_dir,
    )


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


def build_model_config(args: GlobalInferenceArgs, model_name: str, sample_templates: bool):
    config = model_config(model_name)
    config.data.common.max_recycling_iters = args.max_recycling_iters
    config.globals.max_recycling_iters = args.max_recycling_iters
    config.data.predict.num_ensembles = args.num_ensembles
    if sample_templates:
        # enable template samples for diversity
        config.data.predict.subsample_templates = True
    # faster prediction with large chunk
    config.globals.chunk_size = 128
    return config


def build_model(config, param_path: str, model_device: str, bf16: bool):
    model = AlphaFold(config)
    print("start to load params {}".format(param_path))
    state_dict = torch.load(param_path)["ema"]["params"]
    state_dict = {".".join(k.split(".")[1:]): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model = model.to(model_device)
    model.eval()
    if bf16:
        model.bfloat16()
    return model


def get_job_fn_postfix(
    sample_templates: bool, use_uniprot: bool, is_multimer: bool, max_recycling_iters: int, num_ensembles: int
) -> str:
    name_postfix = ""
    if sample_templates:
        name_postfix += "_st"
    if not is_multimer and use_uniprot:
        name_postfix += "_uni"
    name_postfix += "_r" + str(max_recycling_iters)
    if num_ensembles != 2:
        name_postfix += "_e" + str(num_ensembles)
    return name_postfix


def run_main_loop(configs: Dict[str, Any], features_per_seq: Dict[str, List[Any]], output_dirs: List[str], standard_args: StandardArgs):
    print("start to predict {}".format(args.target_name))
    plddts = {}
    ptms = {}
    for model_name, model_config in configs.items():
        print(f"working on model {model_name}")
        is_multimer = model_config.model.is_multimer
        model = build_model(model_config, args.param_path, args.model_device, args.bf16)
        for batch, output_dir in zip(features_per_seq[model_name], output_dirs):
            for seed in range(args.times):
                cur_seed = hash((args.data_random_seed, seed)) % 100000
                seq_len = batch["aatype"].shape[-1]
                model.globals.chunk_size = automatic_chunk_size(seq_len)

                with torch.no_grad():
                    batch = {k: torch.as_tensor(v, device=args.model_device) for k, v in batch.items()}
                    shapes = {k: v.shape for k, v in batch.items()}
                    print(shapes)
                    t = time.perf_counter()
                    raw_out = model(batch)
                    print(f"Inference time: {time.perf_counter() - t}")

                def to_float(x):
                    if x.dtype == torch.bfloat16 or x.dtype == torch.half:
                        return x.float()
                    else:
                        return x

                if not args.save_raw_output:
                    score = ["plddt", "ptm", "iptm", "iptm+ptm"]
                    out = {k: v for k, v in raw_out.items() if k.startswith("final_") or k in score}
                else:
                    out = raw_out
                del raw_out
                # Toss out the recycling dimensions --- we don't need them anymore
                batch = tensor_tree_map(lambda t: t[-1, 0, ...], batch)
                batch = tensor_tree_map(to_float, batch)
                out = tensor_tree_map(lambda t: t[0, ...], out)
                out = tensor_tree_map(to_float, out)
                batch = tensor_tree_map(lambda x: np.array(x.cpu()), batch)
                out = tensor_tree_map(lambda x: np.array(x.cpu()), out)

                plddt = out["plddt"]
                mean_plddt = np.mean(plddt)
                plddt_b_factors = np.repeat(plddt[..., None], residue_constants.atom_type_num, axis=-1)
                # TODO: , may need to reorder chains, based on entity_ids
                cur_protein = protein.from_prediction(features=batch, result=out, b_factors=plddt_b_factors)
                name_postfix = get_job_fn_postfix(
                    standard_args.sample_templates, standard_args.use_uniprot, is_multimer, args.max_recycling_iters, args.num_ensembles
                )
                cur_save_name = f"{args.model_name}_{name_postfix}_{cur_seed}{name_postfix}"
                plddts[cur_save_name] = str(mean_plddt)
                if is_multimer:
                    ptms[cur_save_name] = str(np.mean(out["iptm+ptm"]))

                with open(os.path.join(output_dir, cur_save_name + ".pdb"), "w") as f:
                    f.write(protein.to_pdb(cur_protein))
                if args.save_raw_output:
                    with gzip.open(os.path.join(output_dir, cur_save_name + "_outputs.pkl.gz"), "wb") as f:
                        pickle.dump(out, f)
                del out
        del model
    return plddts, ptms

def run_nomsa_main(args: GlobalInferenceArgs, nomsa_args: NoMSAArgs):
    configs = {x: build_model_config(args, x, False) for x in args.model_names}
    features = {
        x: [
            create_nomsa_features(
                configs[x],
                sequence,
                args.data_random_seed,
                is_multimer=configs[x].model.is_multimer,
            )
            for sequence in nomsa_args.sequences
        ]
        for x in args.model_names
    }
    output_dirs = [str(Path(args.output_dir)/x) for x in args.target_names]
    plddts, ptms = run_main_loop(configs, features, output_dirs, StandardArgs())
    # print("plddts", plddts)
    plddt_fname = Path(args.output_dir) / "plddt.json"
    json.dump(plddts, open(plddt_fname, "w"), indent=2)
    if ptms:
        ptm_fname = Path(args.output_dir) / "ptm.json"
        json.dump(ptms, open(ptm_fname, "w"), indent=2)


def run_standard_main(args: GlobalInferenceArgs, standard_args: StandardArgs):
    configs = {x: build_model_config(args, x, standard_args.sample_templates) for x in args.model_names}
    features = {
        x: [
            load_feature_for_one_target(
                configs[x],
                data_dir,
                args.data_random_seed,
                is_multimer=configs[x].model.is_multimer,
                use_uniprot=StandardArgs.use_uniprot,
            )
            for data_dir in args.data_dirs
        ]
        for x in args.model_names
    }
    output_dirs = [str(Path(args.output_dir)/x) for x in args.target_names]
    run_main_loop(configs, features, output_dirs, standard_args)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_arguments(GlobalInferenceArgs, dest="prog")
    args = parser.parse_args()
    prog: GlobalInferenceArgs = args.prog
    print(prog)
    print(type(prog))
    print(type(prog.subcommand))
    if type(prog.subcommand) == StandardArgs:
        run_standard_main(prog, prog.subcommand)
    elif type(prog.subcommand) == NoMSAArgs:
        run_nomsa_main(prog, prog.subcommand)
    # parser.add_arguments(GlobalInferenceArgs, dest="prog")

    # add_argparse_global_args(parser)
    # main_subparser_name = "subparser_name"
    # subparsers = parser.add_subparsers(dest=main_subparser_name)
    # subparsers.add_parser("standard")
    # no_msa_parser = subparsers.add_parser("no_msa")
    # parser.add_argument("--sequences", required=True, nargs="+", help="Sequences (if multisequence use : to separate sequences)")
    #
    # args = parser.parse_args()
    #
    # if args.model_device == "cpu" and torch.cuda.is_available():
    #     logging.warning(
    #         """The model is being run on CPU. Consider specifying
    #         --model_device for better performance"""
    #     )
    #
    # main(args)
