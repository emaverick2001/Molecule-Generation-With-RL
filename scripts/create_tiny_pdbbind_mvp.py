# scripts/create_tiny_pdbbind_mvp.py

from pathlib import Path


COMPLEX_IDS = ["1abc", "2xyz", "3def", "4ghi", "5jkl"]


FAKE_PROTEIN_PDB = """HEADER    TINY MVP PROTEIN
ATOM      1  N   ALA A   1      11.104  13.207   8.678  1.00 20.00           N
ATOM      2  CA  ALA A   1      12.560  13.300   8.400  1.00 20.00           C
ATOM      3  C   ALA A   1      13.100  12.000   7.800  1.00 20.00           C
ATOM      4  O   ALA A   1      12.500  10.950   7.900  1.00 20.00           O
END
"""


FAKE_LIGAND_SDF = """TinyLigand
  MVP

  3  2  0  0  0  0            999 V2000
    0.0000    0.0000    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
    1.5000    0.0000    0.0000 O   0  0  0  0  0  0  0  0  0  0  0  0
    0.0000    1.5000    0.0000 N   0  0  0  0  0  0  0  0  0  0  0  0
  1  2  1  0
  1  3  1  0
M  END
$$$$
"""


def main() -> None:
    raw_root = Path("data/raw/pdbbind")

    for complex_id in COMPLEX_IDS:
        complex_dir = raw_root / complex_id
        complex_dir.mkdir(parents=True, exist_ok=True)

        (complex_dir / "protein.pdb").write_text(FAKE_PROTEIN_PDB, encoding="utf-8")
        (complex_dir / "ligand.sdf").write_text(FAKE_LIGAND_SDF, encoding="utf-8")
        (complex_dir / "ligand_gt.sdf").write_text(FAKE_LIGAND_SDF, encoding="utf-8")

    print(f"Created tiny MVP PDBBind-style dataset with {len(COMPLEX_IDS)} complexes.")


if __name__ == "__main__":
    main()