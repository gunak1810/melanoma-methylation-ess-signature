import gseapy as gp
import logging

def get_msigdb_sets(category="H"):
    """
    Fetch gene sets from MSigDB using gseapy.
    Category "H" is Hallmark. 
    Category "C2" includes KEGG and REACTOME.
    Category "C5" includes GO.
    """
    try:
        # Get names of libraries
        libraries = gp.get_library_name()
        
        # Map our categories to gseapy library names
        lib_map = {
            "H": "MSigDB_Hallmark_2020",
            "KEGG": "KEGG_2021_Human",
            "REACTOME": "Reactome_2022",
            "GO_BP": "GO_Biological_Process_2023"
        }
        
        lib_name = lib_map.get(category, category)
        
        if lib_name not in libraries:
            logging.warning(f"Library {lib_name} might not be in gseapy. Trying enrichr...")
            
        enr = gp.get_library(name=lib_name, organism='Human')
        return enr
    except Exception as e:
        logging.error(f"Error loading gene sets for {category}: {e}")
        return {}

def get_stemness_markers():
    """
    Top stemness-associated genes (Malta et al. 2018).
    Positive = stem-like, Negative = differentiated.
    """
    return {
        "positive": [
            "HMGA2", "SOX2", "POU5F1", "NANOG", "LIN28A", "LIN28B", "SALL4", 
            "KLF4", "MYC", "TERT", "DNMT3B", "EZH2", "BMI1", "KDM5B", "ALDH1A1", 
            "CD44", "NES", "PROM1", "THY1"
        ],
        "negative": [
            "MITF", "TYR", "DCT", "PMEL", "MLANA"
        ]
    }

def get_melanoma_phenotype_genes():
    """Melanoma differentiation / phenotype switching genes."""
    return {
        "melanocytic": ["MITF", "TYR", "DCT", "PMEL", "MLANA", "TYRP1", 
                        "SOX10", "PAX3", "SLC45A2", "GPR143", "OCA2"],
        "invasive": ["AXL", "WNT5A", "ZEB1", "ZEB2", "TWIST1", "NGFR", 
                     "SERPINE1", "FN1", "TGFBI", "JUN", "FOSL1"],
        "ncsc": ["NGFR", "SOX10", "FOXD3", "RXRG", "AQP1"],
        "transitory": ["SOX9", "SOX10", "MITF"],
        "starved": ["NGFR", "LOXL2", "AXL", "HIF1A", "BNIP3"]
    }

def get_checkpoint_genes():
    """Immune checkpoint genes."""
    return [
        "CD274", "PDCD1", "CTLA4", "LAG3", "HAVCR2", "TIGIT", "IDO1", 
        "SIGLEC15", "CD276", "VSIR", "VTCN1", "ICOS", "TNFRSF9", 
        "TNFRSF4", "TNFRSF18"
    ]
