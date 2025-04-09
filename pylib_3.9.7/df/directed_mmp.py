"""
Specialised MMP Search.  Takes a set of structures, a query mol file or
SMARTS string and a list of atoms from the query structure.  Finds
all MMPs where the query matches and the only difference is between the
query atoms list.  For example,  if the query structure is
c1ncccc1CCc1ccccc1 and the atom list is 6, 7 (the ethylene linker)
then c1ncccc1CCc1ccccc1 and c1ncccc1Oc1ccccc1 would be a matched
pair, as would c1ncccc1CCc1cc(F)ccc1 and c1ncccc1CCCc1cc(F)ccc1
but c1ncccc1CCc1ccccc1 and c1ncccc1CCCc1cc(F)ccc1 wouldn't.
"""

import argparse
import gzip
import sys

from collections import deque, namedtuple
from pathlib import Path
from typing import Optional, Union

from rdkit import Chem, rdBase, Geometry
from rdkit.Chem import AllChem, rdMolAlign, rdqueries, rdDepictor, rdmolops, rdFMCS
from rdkit.Chem.Draw import rdMolDraw2D
NUMBERS = [str(i) for i in range(999)]
Fragments = namedtuple('Fragments',['mol','remainder','replaced','remainder_smi','replaced_smi','rowIdx'])

def parse_args(cli_args: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Project Matched Molecular" " Pairs")
    parser.add_argument(
        "-I",
        "--input-file",
        dest="input_file",
        required=True,
        help="Name of input structure file.",
    )
    parser.add_argument(
        "-O",
        "--output-file",
        dest="output_file",
        required=True,
        help="Name of output structure file for pairs.",
    )
    parser.add_argument(
        "-S",
        "--query-smarts",
        dest="query_smarts",
        help="SMARTS string defining the search substructure.",
    )
    parser.add_argument(
        "--atoms",
        nargs="*",
        dest="atoms",
        type=int,
        required=True,
        help="List of atom numbers in query substructure that"
        " can be replaced in first molecule in pair to"
        " form second molecule.",
    )

    args = parser.parse_args(cli_args)
    return args


def create_mol_supplier(
    infile,
) -> Union[Chem.ForwardSDMolSupplier, Chem.SmilesMolSupplier, None]:
    """

    Args:
        infile (str): must end .smi, .sdf or .sdf.gz

    Returns:
        ForwardSDMolSupplier, SmilesMolSupplier or None
    """
    inpath = Path(infile)
    sfx = inpath.suffix
    gzipped = False
    if sfx == ".gz":
        suffixes = inpath.suffixes
        gzipped = True
        sfx = suffixes[-2]

    if sfx != ".smi" and sfx != ".sdf" and sfx != ".mol":
        print(
            f"ERROR : input must be a SMILES, SDF, MOL or gzipped SDF"
            f" or MOL, you gave {infile}  Aborting."
        )
        return None

    if sfx == ".smi":
        return Chem.SmilesMolSupplier(infile, titleLine=False)
    else:
        try:
            if gzipped:
                inf = gzip.open(infile)
                return Chem.ForwardSDMolSupplier(inf)
            else:
                return Chem.ForwardSDMolSupplier(infile)
        except (OSError, FileNotFoundError):
            print(f"ERROR : failed to open file {infile}.  Not Found.")
            return None


def create_mol_writer(outfile) -> Union[Chem.SDWriter, Chem.SmilesWriter, None]:
    """

    Args:
        infile (str): must end .smi, .sdf or .sdf.gz

    Returns:
        SDMolWriter, SmilesMolWriter or None
    """
    outpath = Path(outfile)
    sfx = outpath.suffix

    if sfx != ".smi" and sfx != ".sdf":
        print(
            f"ERROR : output must be a SMILES or SDF"
            f" or MOL, you gave {outfile}  Aborting."
        )
        return None

    if sfx == ".smi":
        return Chem.SmilesWriter(outfile, titleLine=False)
    else:
        return Chem.SDWriter(outfile)


def read_input_file(input_file: str) -> list[Chem.Mol]:
    input_mols = []
    suppl = create_mol_supplier(input_file)
    for mol in suppl:
        if mol is not None and mol.GetNumAtoms() > 0:
            if not mol.GetNumConformers() or mol.GetConformer().Is3D():
                rdDepictor.Compute2DCoords(mol)
            input_mols.append(mol)

    return input_mols


def remove_atom_num_0_atoms(
    mol: Chem.Mol, atom_idxs: Optional[list[int]] = None
) -> Chem.Mol:
    """
    Remove any atoms with an atomic number of 0, which can interfere
    with the R Group matching process.  If atom_idxs is not None, only
    remove those that are connected to an atom mentioned in it.
    """
    atoms_to_go = []
    if atom_idxs is None:
        atom_idxs_set = set()
    else:
        atom_idxs_set = set(atom_idxs)

    q0 = rdqueries.AtomNumEqualsQueryAtom(0)
    if atom_idxs is not None:
        for atom in mol.GetAtomsMatchingQuery(q0):
            if atom.GetIdx() not in atom_idxs_set:
                for nbr in atom.GetNeighbors():
                    if nbr.GetIdx() in atom_idxs_set:
                        atoms_to_go.append(atom)
    else:
        for atom in mol.GetAtomsMatchingQuery(q0):
            atoms_to_go.append(atom)
    mol_edit = Chem.RWMol(mol)
    mol_edit.BeginBatchEdit()
    for atg in atoms_to_go:
        mol_edit.RemoveAtom(atg.GetIdx())
    mol_edit.CommitBatchEdit()

    return mol_edit


def split_molecule(
    mol: Chem.Mol, ats_to_go: list[int], bonds_to_go: dict[int, int]
) -> tuple[Optional[Chem.Mol], Optional[Chem.Mol]]:
    """
    Split a molecule into two molecules based on bonds and atom
    indices.  Returns two molecules - the bits left over and the
    bit that was ats_to_go.  bonds_to_go is a dict mapping the
    bonds that are to go with the labels of the R Groups to
    be left.  If the split throws an AtomKekulizeException,
    returns None, None.
    Args:
        mol: the molecule to be fragmented.
        ats_to_go:
        bonds_to_go:

    Returns:
        [Chem.Mol, Chem.Mol]
    """
    bonds = []
    labels = []
    for bond, label in bonds_to_go.items():
        bonds.append(bond)
        labels.append((label, label))

    frag_mol = rdmolops.FragmentOnBonds(mol, bonds, dummyLabels=labels)

    # Split the fragments into the bit that was in ats_to_go, which
    # goes into remainder, and the bits that are r groups that go
    # into mol_edit.  An exception means it's a bad match, so don't
    # go any further.
    frags_we_want = []
    remainder = None
    try:
        all_frags = rdmolops.GetMolFrags(frag_mol, asMols=True)
    except Chem.rdchem.AtomKekulizeException:
        return None, None

    for frag in all_frags:
        # if the remainder has yet to be identified, see if it's this one
        if remainder is None:
            for atom in frag.GetAtoms():
                try:
                    orig_idx = int(atom.GetProp("OrigIdx"))
                    if orig_idx in ats_to_go:
                        remainder = frag
                        break
                except KeyError:
                    # WildCard atoms won't have an OrigIdx
                    pass
        # if this is not the remainder, get the isotope number of the
        # attachment point and set it as RGroupNum for all atoms in the
        # fragment and save the frag to go into mol_edit.
        if remainder != frag:
            frags_we_want.append(frag)
            isotope_num = -1
            for atom in frag.GetAtoms():
                if atom.GetAtomicNum() == 0:
                    isotope_num = atom.GetIsotope()
                    break
            isotope_num = str(isotope_num)
            for frag_atom in frag.GetAtoms():
                if not frag_atom.HasProp("RGroupNum"):
                    frag_atom.SetProp("RGroupNum", isotope_num)

    mol_edit = frags_we_want[0]
    for frag in frags_we_want[1:]:
        mol_edit = rdmolops.CombineMols(mol_edit, frag)

    return mol_edit, remainder


def label_atoms_with_idx(mol: Chem.Mol) -> None:
    for i, atom in enumerate(mol.GetAtoms()):
        try:
            atom.SetProp("OrigIdx", NUMBERS[i])
        except IndexError:
            atom.SetProp("OrigIdx", str(i))
            atom.SetProp("OrigIdx", str(i))


def label_atoms_with_r_group(mol: Chem.Mol) -> None:
    # Label the query atoms with the R Group they represent, taking
    # it from the isotope number.
    all_frags = rdmolops.GetMolFrags(mol)
    for frag in all_frags:
        isotope_num = -1
        for frag_atom in frag:
            atom = mol.GetAtomWithIdx(frag_atom)
            if atom.GetAtomicNum() == 0:
                isotope_num = atom.GetIsotope()
                break
        if isotope_num != -1:
            isotope_num_str = NUMBERS[isotope_num]
            for frag_atom in frag:
                atom = mol.GetAtomWithIdx(frag_atom)
                atom.SetProp("RGroupNum", isotope_num_str)


def make_rgroups_query_mol(mol: Chem.Mol) -> Chem.Mol:
    # Turn the r_groups_mol into a query mol and set the atoms so they
    # must match the number of explicit connections.  This will stop
    # the R Groups matching inside a larger RGroup e.g. [*:1]C(C)C will
    # only match the isopropyl in CC1CC(C(C)C)CCC1 and not the one
    # hidden in the methylated ring.  Other properties, such as in-ring
    # could be added as well to tie it down further, but for now there
    # doesn't seem to be a need.
    if hasattr(rdqueries, "ReplaceAtomWithQueryAtom"):
        r_groups_query = Chem.Mol(mol)
        for atom in r_groups_query.GetAtoms():
            qatom = rdqueries.ReplaceAtomWithQueryAtom(r_groups_query, atom)
            qa = rdqueries.ExplicitDegreeEqualsQueryAtom(atom.GetDegree())
            qatom.ExpandQuery(qa, Chem.CompositeQueryType.COMPOSITE_AND)
    else:
        r_groups_query = Chem.MolFromSmarts(Chem.MolToSmiles(mol))
        for atom in r_groups_query.GetAtoms():
            qa = rdqueries.ExplicitDegreeEqualsQueryAtom(atom.GetDegree())
            atom.ExpandQuery(qa, Chem.CompositeQueryType.COMPOSITE_AND)

    return r_groups_query


def check_for_pair_via_smarts(
    ref_mol: Chem.Mol,
    target_mol: Chem.Mol,
    r_groups_mol: Chem.Mol,
) -> tuple[Optional[Chem.Mol], Optional[Chem.Mol]]:
    """
    See if target_mol is a directed MMP with ref_mol.  If it is, return
    the bit of parent_mol that is leftover when the r_groups_mol pieces
    are removed from it.
    Args:
        ref_mol (Chem.Mol): the first molecule under consideration
        target_mol (Chem.Mol): the molecule which could be a pair with
                               ref_mol
        r_groups_mol (Chem.Mol): the parts of ref_mol after the query
                                 bit was removed.

    Returns:
        Chem.Mol
    """

    debug_str = f"check pair --->\n{Chem.MolToSmiles(ref_mol)}\n{Chem.MolToSmiles(target_mol)}\n{Chem.MolToSmiles(r_groups_mol)}"
    r_groups_query = make_rgroups_query_mol(r_groups_mol)
    label_atoms_with_r_group(r_groups_query)

    # The wild cards aren't helpful any more.  The Degree test that has
    # been added to r_groups_query
    # makes them redundant, and they can cause bad results.  For
    # example if the R Groups are ClC* and BrC* and the target
    # molecule is ClCCCBr this won't match the 2 R Groups with the
    # wildcards since there is only 1 linker atom.
    plain_r_groups = remove_atom_num_0_atoms(r_groups_query)

    label_atoms_with_idx(target_mol)

    bits = None
    remainder_mol = None
    for match in target_mol.GetSubstructMatches(plain_r_groups):
        match_set = set(match)
        target_cp = Chem.Mol(target_mol)
        bonds_to_go = {}
        ats_to_go = set([atom.GetIdx() for atom in target_cp.GetAtoms()])
        for i, ma in enumerate(match):
            rgroup = plain_r_groups.GetAtomWithIdx(i).GetProp("RGroupNum")
            atom = target_cp.GetAtomWithIdx(ma)
            atom.SetProp("RGroupNum", rgroup)
            ats_to_go.remove(ma)
            for nbr in atom.GetNeighbors():
                if nbr.GetIdx() not in match_set:
                    bond = target_cp.GetBondBetweenAtoms(nbr.GetIdx(), ma)
                    bonds_to_go[bond.GetIdx()] = int(rgroup)
        bits, remainder_mol = split_molecule(target_cp, list(ats_to_go), bonds_to_go)
        if bits is None:
            continue

        # The bits should be the same as r_groups_mol.  If they're
        # not, it's not a valid solution.
        if Chem.MolToSmiles(bits) != Chem.MolToSmiles(r_groups_mol):
            remainder_mol = None
            continue

        if remainder_mol is not None:
            # Align the target_mol (not the target_cp) onto the ref_mol
            # for depiction if necessary. Sets props on target_mol atoms
            # to map to atoms in ref_mol, but doesn't do the final
            # alignment.
            #align_r_groups(ref_mol, target_mol, plain_r_groups)
            break

    return bits, remainder_mol


def get_bonds_between_atoms(mol: Chem.Mol, at_list: list[int]) -> list[int]:
    bonds = []
    for ta1 in at_list:
        for ta2 in at_list:
            if ta1 > ta2:
                continue
            bond = mol.GetBondBetweenAtoms(ta1, ta2)
            if bond is not None:
                bonds.append(bond.GetIdx())

    return bonds


def add_highlights_to_mol(
    mol: Chem.Mol, atom_list: list[int], bond_list: list[int]
) -> None:
    highlight_colour = "#14aadb"
    atom_highs = " ".join([str(a + 1) for a in atom_list])
    bond_highs = " ".join([str(b + 1) for b in bond_list])
#    high_str = f"COLOR {highlight_colour}\nATOMS {atom_highs}\nBONDS {bond_highs}"
    high_str = f"COLOR {highlight_colour}\nATOMS\nBONDS {bond_highs}"
    mol.SetProp("Renderer_Highlight", high_str)


def highlight_replaced(mol: Chem.Mol, fragment: Chem.Mol) -> None:
    atoms = []
    orig_atoms = []
    for atom in fragment.GetAtoms():
        if atom.HasProp("OrigIdx"):
            orig_atoms.append(int(atom.GetProp("OrigIdx")))

    bonds = get_bonds_between_atoms(mol, orig_atoms)

    add_highlights_to_mol(mol, atoms, bonds)


def remove_directed_bit(
    mol: Chem.Mol,
    match: Union[list[int], tuple[int]],
    atom_idxs: list[int]
) -> tuple[Optional[Chem.Mol], Optional[Chem.Mol]]:
    """
    Take the molecule and remove the atoms in match that are in
    atom_idx.  Return 2 new molecules, the bit removed and the bit
    left and the indices of the atoms in the molecule that were
    removed.  If one of the atom_idxs is bonded to an atom not in the
    match list, it's not a valid query so None, None, None is
    returned.  If one of the bits is attached to an atom in the
    wildcard_nbrs list, it is not part of either bit returned.
    Args:
        mol (Chem.Mol): the molecule to be worked on
        match (tuple[int]): the atom indices that the initial query
                            matched
        atom_idxs (list[int]): the indices of the atoms from the
                               initial query that are to go
    Returns:
        [Chem.Mol, Chem.Mol]: two new molecules, the bits left and the
                              bit removed
    """
    match_set = set(match)
    ats_to_go = [match[ai] for ai in atom_idxs]

    #add atoms attached to selected portion that are not part of the query
    current_atoms = list(ats_to_go)
    while len(current_atoms) != 0:
        next_atoms = []
        for at in current_atoms:
            atom = mol.GetAtomWithIdx(at)
            for nbr in atom.GetNeighbors():
                if nbr.GetIdx() not in ats_to_go and nbr.GetIdx() not in match_set:
                    ats_to_go.append(nbr.GetIdx())
                    next_atoms.append(nbr.GetIdx())
        current_atoms = next_atoms

    ats_to_go_set = set(ats_to_go)

    # env_bonds will be the bonds to go that are from an atom in
    # ats_to_go to one in match i.e. they're in the environment of
    # atom_idxs/ats_to_go.  Other_bonds are bonds to an atom in
    # ats_to_go that aren't in the environment.  It's important that
    # the former are numbered consistently.
    env_bonds = []
    other_bonds = []
    for at in ats_to_go:
        atom = mol.GetAtomWithIdx(at)
        for nbr in atom.GetNeighbors():
            bond = mol.GetBondBetweenAtoms(at, nbr.GetIdx()).GetIdx()
            if nbr.GetIdx() not in ats_to_go_set and nbr.GetIdx() in match_set:
                env_bonds.append(bond)
            elif nbr.GetIdx() not in match_set:
                other_bonds.append(bond)
    bonds_to_go = {}
    i = 0
    for i, bond in enumerate(env_bonds, 1):
        bonds_to_go[bond] = i

    if not bonds_to_go:
        return None, None
    remainder, replaced = split_molecule(mol, ats_to_go, bonds_to_go)

    return remainder, replaced


def normalize_bond_lengths(mol):

    meanBondLen = rdMolDraw2D.MeanBondLength(mol)
    if abs(1.5 - meanBondLen) > 0.01:
        conf = mol.GetConformer()
        scale = 1.5/meanBondLen
        for atom in mol.GetAtoms():
            pos = conf.GetAtomPosition(atom.GetIdx())
            pos.x *= scale
            pos.y *= scale
            conf.SetAtomPosition(atom.GetIdx(),pos)


def find_pairs(
    input_mols: list[Chem.Mol], query: Chem.Mol, atom_idxs: list[int]
) -> list[tuple[Chem.Mol, Chem.Mol, Chem.Mol, Chem.Mol]]:
    """
    Find the pairs for each molecule in input_mols.  Returns the
    molecules in each pair, plus the bit of the 1st molecule that was
    replaced and the bit of the 2nd molecule that's the equivalent of
    the atom_idxs atoms in the query in the 1st molecule that replaced
    it.
    Args:
        input_mols:
        query:
        atom_idxs: a list of integers representing the atoms in the query
                   that are to be replaced in the matched pairs

    Returns:
        list[tuple[Chem.Mol, Chem.Mol, Chem.Mol, Chem.Mol]]
    """

    if query.GetNumConformers() == 0:
        rdDepictor.Compute2DCoords(query)
    normalize_bond_lengths(query)

    # find all query matches, depict using query as template, split input into fragments using query
    aligned = set()
    fragged = []
    no_substruct_match = []
    for mol in input_mols:
        if mol is None:
            continue

        label_atoms_with_idx(mol)
        matches = mol.GetSubstructMatches(query, maxMatches=1)
        if matches:
            rdDepictor.GenerateDepictionMatching2DStructure(mol, query) #use query as depiction template
            rowIdx = int(mol.GetProp("RowNum"))
            aligned.add(rowIdx)
            remainder, replaced = remove_directed_bit(mol, matches[0], atom_idxs)
            if remainder and replaced:
                highlight_replaced(mol, replaced)
                replaced_smi = Chem.MolToSmiles(replaced)
                remainder_smi = Chem.MolToSmiles(remainder)
                fragged.append(Fragments(mol, remainder, replaced, remainder_smi, replaced_smi, rowIdx))
        elif not mol.GetNumConformers() or mol.GetConformer().Is3D():
            rdDepictor.Compute2DCoords(mol)
            no_substruct_match.append(mol)

    #create maps of replaced and remainder fragments
    uniq_replaced = {}
    uniq_remainder = {}
    for frag in fragged:
        if frag.replaced_smi in uniq_replaced:
            uniq_replaced[frag.replaced_smi].append(frag)
        else:
            uniq_replaced[frag.replaced_smi] = [frag]
        if frag.remainder_smi in uniq_remainder:
            uniq_remainder[frag.remainder_smi].append(frag)
        else:
            uniq_remainder[frag.remainder_smi] = [frag]

    #for each unique replaced, iterate over all identical remainders that have a different replaced and create pairs
    pairs = []
    for key,val in uniq_replaced.items():
        for repl_frag in val:
            if repl_frag.remainder_smi in uniq_remainder:
                for rem_frag in uniq_remainder[repl_frag.remainder_smi]:
                    if repl_frag.replaced_smi != rem_frag.replaced_smi:
                        lhs = Chem.Mol(repl_frag.mol)
                        rhs = Chem.Mol(rem_frag.mol)
                        pairs.append((lhs, rhs, repl_frag.replaced, rem_frag.replaced))

    #find non-symmetric matches
    if len(no_substruct_match) != 0:
        for key,val in uniq_remainder.items():
            for target_mol in no_substruct_match:
                remainder, replaced = check_for_pair_via_smarts(val[0].mol, target_mol, val[0].remainder)
                if remainder is not None and replaced is not None:
                    rhs = target_mol
                    highlight_replaced(rhs, replaced)
                    mcs = rdFMCS.FindMCS([val[0].mol, target_mol], maximizeBonds=True,threshold=1.0,timeout=1,matchValences=False,ringMatchesRingOnly=True,completeRingsOnly=True)
                    if mcs and mcs.numAtoms > 3:
                        rdDepictor.GenerateDepictionMatching2DStructure(target_mol, val[0].mol, refPatt=mcs.queryMol)

                    replaced_smi = Chem.MolToSmiles(replaced)
                    for frag in val:
                        if replaced_smi != frag.replaced_smi:
                            lhs = frag.mol
                            pairs.append((lhs, rhs, frag.replaced, replaced))

    return pairs


def main(cli_args: list[str]) -> bool:
    print(f"RDKit version : {rdBase.rdkitVersion}")
    rdDepictor.SetPreferCoordGen(True)
    args = parse_args(cli_args)
    if args.query_smarts is not None:
        query = Chem.MolFromSmarts(args.query_smarts)

    input_mols = read_input_file(args.input_file)
    pairs = find_pairs(input_mols, query, args.atoms)
    for p in pairs:
        n0 = p[0].GetProp("_Name")
        n1 = p[1].GetProp("_Name")

    writer = create_mol_writer(args.output_file)
    for p in pairs:
        writer.write(p[0])
        writer.write(p[1])
        writer.write(p[2])

    return True


if __name__ == "__main__":
    sys.exit(not main(sys.argv[1:]))
