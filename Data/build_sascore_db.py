#!/usr/bin/env python
#
# File to build a database of fragment scores for the Synthetic
# Accessibility score of Ertl and Schuffenhauer
# Journal of Cheminformatics 1:8 (2009)
# http://www.jcheminf.com/content/1/1/8
#
# It is what is used to build surechembl_scores.pkl.gz
import argparse
import gzip
import math
import pickle

try:
    import seaborn.objects as so
    HAVE_SEABORN = True
except ModuleNotFoundError:
    HAVE_SEABORN = False
import sys

from collections import defaultdict
from pathlib import Path
from typing import Optional, Union

try:
    from ruse.rdkit.rdkit_utils import standardize_mol
    HAVE_STANDARDIZE = True
except ModuleNotFoundError:
    print("Couldn't import standardize_mol so won't be able to standardize molecules.")
    HAVE_STANDARDIZE = False
    
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.MolStandardize import rdMolStandardize


def parse_args(cli_args: list[str]) -> argparse.Namespace:
    """
        Use argparse module to parse the command line arguments.

        Returns:
            (Namespace object): Parsed arguments in argparse Namespace
                                object.
        """
    parser = argparse.ArgumentParser(description='Generate fragment scores'
                                                 ' data for Synthetic'
                                                 ' Accessibility (SA) Score.')
    parser.add_argument('-S', '--score-file', dest='score_file',
                        required=True,
                        help='Name of the output scores file.')
    in_file_args = parser.add_mutually_exclusive_group(required=True)
    in_file_args.add_argument('-I', '--input-file', dest='input_file',
                              help='Name of molecule file containing input'
                                   ' molecules to be processed.')
    in_file_args.add_argument('--input-freqs-pickle', dest='in_freq_pickle',
                              help='Name of previously generated frequencies'
                                   ' pickle file.')
    parser.add_argument('-P', '--plot-file', dest='plot_file',
                        default='dist.svg',
                        help='Name of file for plot of distribution of'
                             ' frequencies. Must use an extension recognised by'
                             ' matplotlib.  Default=%(default)s.')
    parser.add_argument('--output-freqs-pickle', dest='out_freq_pickle',
                        help='If this is given, the raw frequencies from the'
                             ' molecules will be pickled to this file.')
    parser.add_argument('--skip-standardization', dest='skip_stand',
                        action='store_true',
                        help='Skips the molecule standardization step.')

    args = parser.parse_args(cli_args)
    return args


def create_mol_supplier(infile) -> Union[Chem.ForwardSDMolSupplier, Chem.SmilesMolSupplier, None]:
    """

    Args:
        infile (str): must end .smi, .sdf or .sdf.gz

    Returns:
        ForwardSDMolSupplier or None
    """
    inpath = Path(infile)
    sfx = inpath.suffix
    gzipped = False
    if sfx == '.gz':
        suffixes = inpath.suffixes
        gzipped = True
        sfx = suffixes[-2]

    if sfx != '.smi' and sfx != '.sdf' and sfx != '.mol':
        print(f'ERROR : input must be a SMILES, SDF, MOL or gzipped SDF'
              f' or MOL, you gave {infile}  Aborting.')
        return None

    if sfx == '.smi':
        return Chem.SmilesMolSupplier(infile, titleLine=False, sanitize=False)
    else:
        try:
            if gzipped:
                inf = gzip.open(infile)
                return Chem.ForwardSDMolSupplier(inf, sanitize=False)
            else:
                return Chem.ForwardSDMolSupplier(infile, sanitize=False)
        except (OSError, FileNotFoundError):
            print(f'ERROR : failed to open file {infile}.  Not Found.')
            return None


def extractFragments(molfile: str, skip_stand: bool) -> Optional[list[tuple[int, int]]]:
    """
    Read the input file and return a list of frequency counts of
    radius 2 Morgan fingerprints.  Sorted in descending order of
    frequency.  First element in each tuple is the fragment number,
    second is the count.
    If skip_stand is True, doesn't do the molecule standardization step
    which is slow and can be done externally in parallel.
    """
    suppl = create_mol_supplier(molfile)
    if suppl is None:
        return None

    # SureChEMBL has some odd azides that the normalizer doesn't touch.
    fix_azide = AllChem.ReactionFromSmarts('[N-:1]=[N:2]#[N+:3]>>[N+0:1]#[N+:2][N-:3]')
    normer = rdMolStandardize.Normalizer()
    uncharger = rdMolStandardize.Uncharger()
    metal_disconnector = rdMolStandardize.MetalDisconnector()

    freqDict = defaultdict(int)
    mfpgen = rdFingerprintGenerator.GetMorganGenerator(radius=2)

    for i, mol in enumerate(suppl):
        if i and not i % 1000:
            print(f'Done {i} molecules.', flush=True)
        if mol is None or not mol:
            continue
        # print(f'Next mol : {Chem.MolToSmiles(mol)}  {mol.GetProp("_Name")}', flush=True)
        if skip_stand or not HAVE_STANDARDIZE:
            norm_mol = mol
        else:
            norm_mol = standardize_mol(mol, normer, uncharger,
                                       metal_disconnector, fix_azide)
        if norm_mol is None:
            continue

        sfp = mfpgen.GetSparseCountFingerprint(norm_mol)
        for bitId in sfp.GetNonzeroElements():
            freqDict[bitId] += sfp[bitId]

    freqs = list(freqDict.items())
    freqs.sort(key=lambda f: f[1], reverse=True)

    return freqs


def plot_freqs(freqs: list[tuple[int, int]],
               output_file: str) -> None:
    """
    Do a log frequency plot of the data.  Assume it is already sorted.
    """
    y_freqs = [f[1] for f in freqs]
    (
        so.Plot(x=range(len(y_freqs)), y=y_freqs)
        .add(so.Line())
        .scale(y='log')
        .label(x='fragment number', y='log frequency')
        .save(output_file)
    )


def score_fragments(freqs: list[tuple[int, int]]) -> dict[str, list[int]]:
    """
    Calculate the score for each fragment.  This is the "logarithm of
    the ratio between the actual fragment count and the number of
    fragments forming 80% of all fragments in the database"
    Which is a bit ambiguous to my reading.  Reverse-engineering the
    RDKit pickle file of scores shows that it's a log10, not natural
    log, and the denominator is 1338.
    The denominator thus seems to be summing the total frequencies
    and then finding the number of fragments needed to give a sum of
    frequencies that's 80% of that, summing in descending order of
    frequency.  For the 22M compounds of SureChEMBL as of April 2023,
    that gives a denominator of 1336.
    Returns a dict of scores, where the key is a score, and the items
    are lists of fragments with that score.  This is the original
    format used for the pickle file in the original RDKit contrib
    version of this score.  The score is taken to 4 decimal places
    and stringified.
    freqs contains tuples of fragment id and frequency, and is assumed
    already to be sorted in descending frequency.
    """
    tot_freqs = 0
    for f in freqs:
        tot_freqs += f[1]

    tot_freqs_80 = (tot_freqs * 8) // 10
    tf = 0
    for i, freq in enumerate(freqs):
        tf += freq[1]
        if tf < tot_freqs_80:
            freq_80_num = i
    print(f'tot_freqs : {tot_freqs} : tot_freqs_80 : {tot_freqs_80}'
          f'  freq_80_num = {freq_80_num}')

    scores_dict = defaultdict(list)
    for i, freq in enumerate(freqs):
        score = f'{math.log10(freq[1] / freq_80_num):.4f}'.strip()
        scores_dict[score].append(freq[0])

    return scores_dict


def save_scores_dict(scores_dict: dict[str, list[int]], score_file: str) -> None:
    """
    The pickle appears to be list of lists, each one headed by a score
    followed by the fragments with that score.
    """
    out_lists = []
    for score, frags in scores_dict.items():
        out_lists.append([score] + frags)
    # for ol in out_lists:
    #     print(f'{len(ol)} : {ol[:10]}')
    pickle.dump(out_lists, gzip.open(score_file, 'wb'))


def main(cli_args):
    args = parse_args(cli_args)
    if args.input_file is not None:
        freqs = extractFragments(args.input_file, args.skip_stand)
    else:
        print(f'reading frequencies from {args.in_freq_pickle}')
        with open(args.in_freq_pickle, 'rb') as f:
            freqs = pickle.load(f)

    if freqs is None:
        return False

    print(freqs[:10])
    print(freqs[-10:])
    num_1000 = 0
    num_1 = 0
    for f in freqs:
        if f[1] > 1000:
            num_1000 += 1
        if f[1] == 1:
            num_1 += 1
    print(f'num above 1000 : {num_1000}  num 1 : {num_1}')

    print(f'number of fragments : {len(freqs)}')
    if HAVE_SEABORN:
        plot_freqs(freqs, args.plot_file)
    else:
        print(f'No plot file {args.plot_file} because no seaborn found.')
    if args.out_freq_pickle is not None:
        with open(args.out_freq_pickle, 'wb') as f:
            pickle.dump(freqs, f)

    scores_dict = score_fragments(freqs)
    save_scores_dict(scores_dict, args.score_file)

    return True


if __name__ == '__main__':
    sys.exit(not main(sys.argv[1:]))
