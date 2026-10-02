import os
from Bio import Entrez
from Bio import Medline
import pandas as pd
import time
import re
import screening_helper

Entrez.email = "researcher@example.com"

# 2. Search PubMed
query = '''
(
  ("Melanoma"[Mesh] OR "cutaneous melanoma" OR "SKCM")
  AND
  ("DNA Methylation"[Mesh] OR "epigenetic*" OR "methylome")
  AND
  ("Prognosis"[Mesh] OR "prognostic" OR "survival" OR "signature")
)
AND (2010:2026[dp])
'''

print("Searching PubMed...")
handle = Entrez.esearch(
    db="pubmed",
    term=query,
    retmax=10000
)
results = Entrez.read(handle)
pmids = results["IdList"]
print(f"Found {len(pmids)} articles")

if not pmids:
    print("No articles found. Exiting.")
    exit()

# 3 & 4. Extract Title, Authors, Journal, Abstract
print("Downloading and parsing article details in batches...")
articles = []
for start in range(0, len(pmids), 200):
    batch = pmids[start:start+200]
    try:
        handle = Entrez.efetch(
            db="pubmed",
            id=batch,
            rettype="medline",
            retmode="text"
        )
        records = Medline.parse(handle)
        for record in records:
            # Extract DOI from AID or LID
            doi = ""
            aid_list = record.get("AID", [])
            if not isinstance(aid_list, list):
                aid_list = [aid_list]
            for aid in aid_list:
                if "[doi]" in str(aid).lower():
                    doi = str(aid).replace(" [doi]", "").replace("[doi]", "").strip()
                    break
            if not doi:
                lid_list = record.get("LID", [])
                if not isinstance(lid_list, list):
                    lid_list = [lid_list]
                for lid in lid_list:
                    if "[doi]" in str(lid).lower():
                        parts = str(lid).split()
                        for p in parts:
                            if p.startswith("10."):
                                doi = p.strip()
                                break
                        if not doi:
                            doi = str(lid).replace(" [doi]", "").replace("[doi]", "").strip()
                        break

            ad = record.get("AD", "")
            affiliations = "; ".join(ad) if isinstance(ad, list) else str(ad)
            
            articles.append({
                "PMID": record.get("PMID", ""),
                "DOI": doi,
                "Title": record.get("TI", ""),
                "Journal": record.get("JT", ""),
                "Year": record.get("DP", ""),
                "Authors": "; ".join(record.get("AU", [])),
                "Affiliations": affiliations,
                "MeSH_Terms": "; ".join(record.get("MH", [])),
                "Keywords": "; ".join(record.get("OT", [])),
                "Publication_Type": "; ".join(record.get("PT", [])),
                "Abstract": record.get("AB", "")
            })
        time.sleep(0.5) # Be polite to NCBI servers
    except Exception as e:
        print(f"Failed to fetch batch {start}-{start+200}: {e}")

df = pd.DataFrame(articles)

DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")  # set this in your environment; do not hardcode
DEEPSEEK_LIMIT = 100  # Extract top 100
MAX_API_WORKERS = 2

# Run advanced screening
print("Running advanced pre-screening and relevance scoring...")
scored_df = screening_helper.score_dataframe(df)

# Sort by relevance score descending, then by publication year and PMID
def extract_year(dp_str):
    if not dp_str:
        return 0
    match = re.search(r'\b(19|20)\d{2}\b', str(dp_str))
    return int(match.group(0)) if match else 0

scored_df['Year_Int'] = scored_df['Year'].apply(extract_year)
scored_df = scored_df.sort_values(by=['Relevance_Score', 'Year_Int', 'PMID'], ascending=[False, False, False])
scored_df = scored_df.drop(columns=['Year_Int'])

# Run DeepSeek API Proofreading & Screening
print("Starting DeepSeek API proofreading and screening...")
if DEEPSEEK_API_KEY:
    scored_df = screening_helper.score_dataframe_deepseek(
        scored_df, 
        api_key=DEEPSEEK_API_KEY, 
        cache_file="deepseek_cache.json", 
        max_workers=MAX_API_WORKERS, 
        limit=DEEPSEEK_LIMIT
    )
else:
    print("WARNING: DeepSeek API Key not provided. Skipping LLM proofreading.")
    scored_df["DeepSeek_Confidence_Score"] = 0
    scored_df["DeepSeek_Relevance_Score"] = 0
    scored_df["DeepSeek_Decision"] = "Exclude" # fallback
    scored_df["DeepSeek_Targets"] = "None"
    scored_df["DeepSeek_Model"] = "not reported"
    scored_df["DeepSeek_Screening_Type"] = "not reported"
    scored_df["DeepSeek_Sample_Size"] = "not reported"
    scored_df["DeepSeek_Reason"] = scored_df["Automated_Reason"]

scored_df.to_excel("pubmed_results.xlsx", index=False)
print("Saved pubmed_results.xlsx")

# 5. Download Citation File (.nbib)
print("Downloading citation file (.nbib)...")
with open("pubmed_results.nbib", "w", encoding="utf-8") as f:
    for start in range(0, len(pmids), 200):
        batch = pmids[start:start+200]
        try:
            handle = Entrez.efetch(
                db="pubmed",
                id=",".join(batch),
                rettype="medline",
                retmode="text"
            )
            f.write(handle.read())
            time.sleep(0.5) # Be polite to NCBI servers
        except Exception as e:
            print(f"Failed to fetch citation batch {start}-{start+200}: {e}")
print("Saved pubmed_results.nbib")

# 6. Create a Screening Sheet Automatically
print("Creating screening sheet...")
screening_columns = [
    "Title", "Abstract", "Keywords", "MeSH_Terms", "Publication_Type", 
    "Relevance_Score", "Relevance_Category", "Matched_Keywords", "Flags",
    "DeepSeek_Decision", "DeepSeek_Confidence_Score", "DeepSeek_Relevance_Score", 
    "DeepSeek_Targets", "DeepSeek_Model", "DeepSeek_Screening_Type", "DeepSeek_Sample_Size"
]
if "PMID" in scored_df.columns:
    screening_columns.insert(0, "PMID")
if "DOI" in scored_df.columns:
    screening_columns.insert(1, "DOI")

screening_df = scored_df[[col for col in screening_columns if col in scored_df.columns]].copy()

# Use DeepSeek Decision and Reason as primary pre-populated values
screening_df["Decision"] = scored_df["DeepSeek_Decision"]
screening_df["Reason"] = scored_df["DeepSeek_Reason"]

screening_df.to_excel("screening_sheet.xlsx", index=False)
print("Saved screening_sheet.xlsx")

# 7. Get Full Metadata for Meta-analysis
print("Creating meta-extraction template with DeepSeek extracted data (Top 100)...")
# Filter to only include articles that were processed by DeepSeek
extracted_df = scored_df[~scored_df["DeepSeek_Decision"].str.contains("Unscreened", na=False, case=False)].copy()

meta_df = pd.DataFrame({
    "PMID": extracted_df.get("PMID", ""),
    "DOI": extracted_df.get("DOI", ""),
    "Title": extracted_df.get("Title", ""),
    "Author": extracted_df.get("Authors", ""),
    "Affiliations": extracted_df.get("Affiliations", ""),
    "MeSH_Terms": extracted_df.get("MeSH_Terms", ""),
    "Keywords": extracted_df.get("Keywords", ""),
    "Publication_Type": extracted_df.get("Publication_Type", ""),
    "Year": extracted_df.get("Year", ""),
    "Journal": extracted_df.get("Journal", ""),
    "Screening_Type": extracted_df.get("DeepSeek_Screening_Type", ""),
    "Model_System": extracted_df.get("DeepSeek_Model", ""),
    "Targets": extracted_df.get("DeepSeek_Targets", ""),
    "Validation_Method": extracted_df.get("DeepSeek_Validation", "not extracted"),
    "DeepSeek_Decision": extracted_df.get("DeepSeek_Decision", "")
})

meta_df.to_excel("meta_extraction_template.xlsx", index=False)
print("Saved meta_extraction_template.xlsx with extracted metadata")

# 8. Report pre-screening statistics
print("Pre-screening summary:")
high_count = len(scored_df[scored_df["DeepSeek_Decision"] == "Include"])
manual_count = len(scored_df[scored_df["DeepSeek_Decision"] == "Manual review"])
low_count = len(scored_df[scored_df["DeepSeek_Decision"] == "Exclude"])
unscreened_count = len(scored_df[scored_df["DeepSeek_Decision"].str.contains("Unscreened", na=False)])
print(f"  - Include (High confidence): {high_count} articles")
print(f"  - Manual review: {manual_count} articles")
print(f"  - Exclude: {low_count} articles")
if unscreened_count > 0:
    print(f"  - Unscreened by DeepSeek: {unscreened_count} articles")
print("All tasks completed.")
