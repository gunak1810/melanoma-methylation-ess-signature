import re
import pandas as pd
import requests
import json
import time
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

# Compiled regex patterns for efficient search
PATTERNS = {
    "Melanoma": re.compile(r"\bmelanoma\b|\bskcm\b", re.IGNORECASE),
    "Methylation": re.compile(r"\bmethylation\b|\bepigenetic\b|\bmethylome\b", re.IGNORECASE),
    "Prognosis": re.compile(r"\bprognostic\b|\bsurvival\b|\bsignature\b|\bprognosis\b", re.IGNORECASE),
    "Review Flag": re.compile(r"\bsystematic review\b|\bmeta-analysis\b|\bliterature review\b|\breview article\b", re.IGNORECASE),
}

def analyze_article(title, abstract):
    """
    Analyzes an article title and abstract.
    Returns a dictionary of metrics, score, category, recommended decision, and reason.
    """
    title_str = str(title or "").strip()
    abstract_str = str(abstract or "").strip()
    combined_text = title_str + " " + abstract_str
    
    matched_indicators = []
    matched_flags = []
    
    score = 0
    if PATTERNS["Melanoma"].search(combined_text):
        score += 2
        matched_indicators.append("Melanoma")
    if PATTERNS["Methylation"].search(combined_text):
        score += 2
        matched_indicators.append("Methylation")
    if PATTERNS["Prognosis"].search(combined_text):
        score += 2
        matched_indicators.append("Prognosis")
        
    is_review = bool(PATTERNS["Review Flag"].search(combined_text))
    if is_review:
        score -= 2
        matched_flags.append("Review")
        
    if score >= 3 and not is_review:
        category = "High"
        decision = "Include"
        reason = "High relevance: Matched multiple melanoma methylation keywords."
    elif score >= 1 and not is_review:
        category = "Medium"
        decision = "Maybe"
        reason = f"Medium relevance: Mentions {matched_indicators} but might lack specificity."
    else:
        category = "Low"
        decision = "Exclude"
        if is_review:
            reason = "Exclude: Flagged as review/meta-analysis article."
        else:
            reason = "Exclude: Low relevance score."
            
    return {
        "Relevance_Score": score,
        "Relevance_Category": category,
        "Matched_Keywords": ", ".join(matched_indicators) if matched_indicators else "None",
        "Flags": ", ".join(matched_flags) if matched_flags else "None",
        "Suggested_Decision": decision,
        "Automated_Reason": reason
    }

def score_dataframe(df):
    """
    Applies analysis to a dataframe and adds columns.
    """
    # Drop existing columns to prevent duplicate column names
    cols_to_drop = ["Relevance_Score", "Relevance_Category", "Matched_Keywords", "Flags", "Suggested_Decision", "Automated_Reason"]
    df_clean = df.drop(columns=[col for col in cols_to_drop if col in df.columns])

    results = []
    for idx, row in df_clean.iterrows():
        title = row.get("Title", "")
        abstract = row.get("Abstract", "")
        analysis = analyze_article(title, abstract)
        results.append(analysis)
        
    analysis_df = pd.DataFrame(results)
    return pd.concat([df_clean.reset_index(drop=True), analysis_df.reset_index(drop=True)], axis=1)

def analyze_article_deepseek(title, abstract, api_key):
    """
    Analyzes an article title and abstract using DeepSeek API with JSON mode.
    Returns a dictionary of results or None if failed.
    """
    url = "https://api.iceyyy.dev/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    system_prompt = """You are an expert medical researcher performing a systematic review on DNA methylation prognostic signatures in cutaneous melanoma.

Evaluate the article based on these Mandatory Rules:
Rule 1: Must involve Cutaneous Melanoma.
Rule 2: Must evaluate DNA methylation (e.g., Illumina 450K/EPIC) as a prognostic biomarker or signature.
Rule 3: Must be an original research study. Reject reviews and non-research articles.

Return ONLY JSON in this exact format:
{
  "is_original_study": true,
  "is_melanoma": true,
  "is_methylation": true,
  "study_type": "clinical study",
  "targets_identified": ["DNA methylation"],
  "model_system": "human cohort",
  "has_validation": true,
  "should_include": true,
  "confidence": 0.95,
  "reason": "Brief explanation"
}
"""
    user_prompt = f"Title: {title}\n\nAbstract: {abstract}"
    data = {
        "model": "deepseek-v4.1-flash",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "stream": False
    }
    
    max_retries = 5
    backoff = 1.0
    
    for attempt in range(max_retries):
        try:
            response = requests.post(url, headers=headers, json=data, timeout=20)
            if response.status_code == 200:
                res_data = response.json()
                content_str = res_data["choices"][0]["message"]["content"]
                result = json.loads(content_str)
                return result
            elif response.status_code == 429:
                time.sleep(backoff)
                backoff *= 2.0
            else:
                time.sleep(backoff)
                backoff *= 1.5
        except Exception as e:
            time.sleep(backoff)
            backoff *= 1.5
            
    return None

def score_dataframe_deepseek(df, api_key, cache_file="deepseek_cache.json", max_workers=10, limit=None):
    """
    Screens a dataframe of articles using DeepSeek API in parallel, utilizing a local cache file to preserve progress.
    """
    # Drop existing columns to prevent duplicate column names
    cols_to_drop = [
        "DeepSeek_Confidence_Score", "DeepSeek_Relevance_Score", "DeepSeek_Decision", 
        "DeepSeek_Targets", "DeepSeek_Model", "DeepSeek_Screening_Type", 
        "DeepSeek_Sample_Size", "DeepSeek_Reason"
    ]
    df_clean = df.drop(columns=[col for col in cols_to_drop if col in df.columns])

    # Load cache
    cache = {}
    if os.path.exists(cache_file):
        try:
            with open(cache_file, "r", encoding="utf-8") as f:
                cache = json.load(f)
            print(f"Loaded {len(cache)} articles from cache.")
        except Exception as e:
            print(f"Error loading cache: {e}")
            
    # We use lowercase, alphanumeric title as cache key for robust lookup
    def get_cache_key(row):
        title = str(row.get("Title", ""))
        return re.sub(r'[^a-z0-9]', '', title.lower()).strip()
        
    df_to_process = df_clean.copy()
    if limit is not None:
        # Sort by Relevance_Score descending if available, to prioritize sending highly relevant articles to DeepSeek first
        if "Relevance_Score" in df_to_process.columns:
            df_to_process = df_to_process.sort_values(by="Relevance_Score", ascending=False)
        df_to_process = df_to_process.head(limit)
        
    cache_lock = threading.Lock()
    
    # Save cache helper
    def save_cache():
        with cache_lock:
            try:
                with open(cache_file, "w", encoding="utf-8") as f:
                    json.dump(cache, f, indent=2, ensure_ascii=False)
            except Exception as e:
                print(f"Error saving cache: {e}")
                
    # Function for each thread
    def process_row(index, row):
        key = get_cache_key(row)
        
        # Check cache
        with cache_lock:
            if key in cache:
                return index, cache[key]
                
        title = row.get("Title", "")
        abstract = row.get("Abstract", "")
        
        # Call API
        result = analyze_article_deepseek(title, abstract, api_key)
        
        if result is not None:
            with cache_lock:
                cache[key] = result
            return index, result
        else:
            return index, None

    results = {}
    to_fetch = []
    
    # Filter rows that need fetching
    for idx, row in df_to_process.iterrows():
        key = get_cache_key(row)
        if key in cache:
            results[idx] = cache[key]
        else:
            to_fetch.append((idx, row))
            
    print(f"Total unique articles to process: {len(df_to_process)}")
    print(f"Already in cache: {len(results)}. Need to query: {len(to_fetch)}")
    
    if to_fetch:
        print(f"Starting query of {len(to_fetch)} articles with {max_workers} threads...")
        completed_count = 0
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_idx = {executor.submit(process_row, idx, row): idx for idx, row in to_fetch}
            
            for future in as_completed(future_to_idx):
                idx, result = future.result()
                if result is not None:
                    results[idx] = result
                completed_count += 1
                
                # Periodically save cache and show progress
                if completed_count % 10 == 0 or completed_count == len(to_fetch):
                    save_cache()
                    print(f"Progress: {completed_count}/{len(to_fetch)} queries completed...")
                    
    # Integrate results back into the dataframe
    confidence_scores = []
    comparison_scores = []
    decisions = []
    indices_list = []
    reference_standards = []
    extracted_aucs = []
    sample_sizes = []
    reasons = []
    
    for idx, row in df_clean.iterrows():
        key = get_cache_key(row)
        if key in cache:
            res = cache[key]
            
            # --- Parsing ---
            is_original = bool(res.get("is_original_study", False))
            is_crispr = bool(res.get("is_melanoma", False))
            is_immunity = bool(res.get("is_methylation", False))
            scr_type = res.get("study_type", "not reported")
            targets = res.get("targets_identified", [])
            model_sys = res.get("model_system", "not reported")
            has_val = bool(res.get("has_validation", False))
            
            # --- Scoring ---
            conf_score = 0
            if is_original: conf_score += 3
            if is_crispr: conf_score += 4
            if is_immunity: conf_score += 4
            if has_val: conf_score += 3
            
            comp_score = 0
            if is_crispr and is_immunity: comp_score = 5
            
            # --- Decision Logic ---
            must_pass = True
            fail_reason = ""
            
            if not is_original:
                must_pass = False
                fail_reason = "Fails Rule 3: Not an original study."
            elif not is_crispr:
                must_pass = False
                fail_reason = "Fails Rule 1: Not about melanoma."
            elif not is_immunity:
                must_pass = False
                fail_reason = "Fails Rule 2: Not evaluating DNA methylation."
            
            if not must_pass:
                decision = "Exclude"
                final_reason = f"{fail_reason} Score: {conf_score}/14."
            else:
                if conf_score >= 12:
                    decision = "Automatic Include"
                elif conf_score >= 8:
                    decision = "Manual Review"
                else:
                    decision = "Exclude"
                final_reason = f"Passed core rules. Score: {conf_score}/14. " + str(res.get("reason", ""))
                
            confidence_scores.append(conf_score)
            comparison_scores.append(comp_score)
            decisions.append(decision)
            indices_list.append(", ".join([str(x) for x in targets]) if isinstance(targets, list) else str(targets))
            reference_standards.append(str(model_sys))
            extracted_aucs.append(str(scr_type))
            sample_sizes.append("not reported")
            reasons.append(final_reason)
            
        else:
            confidence_scores.append(0)
            comparison_scores.append(0)
            decisions.append("Exclude (Unscreened)")
            indices_list.append("None")
            reference_standards.append("not reported")
            extracted_aucs.append("not reported")
            sample_sizes.append("not reported")
            reasons.append("Article not processed by DeepSeek API (limit reached).")
            
    df_clean["DeepSeek_Confidence_Score"] = confidence_scores
    df_clean["DeepSeek_Relevance_Score"] = comparison_scores
    df_clean["DeepSeek_Decision"] = decisions
    df_clean["DeepSeek_Targets"] = indices_list
    df_clean["DeepSeek_Model"] = reference_standards
    df_clean["DeepSeek_Screening_Type"] = extracted_aucs
    df_clean["DeepSeek_Sample_Size"] = sample_sizes
    df_clean["DeepSeek_Reason"] = reasons
    
    return df_clean
