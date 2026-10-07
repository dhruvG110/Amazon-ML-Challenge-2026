import itertools
import re
import unicodedata

import numpy as np
import pandas as pd
import regex
import streamlit as st
import lightgbm as lgb
from rapidfuzz import fuzz

# ==========================================
# Pick a Source 1 record -> see candidates, scores, final matches
# Run with:  streamlit run explore_app.py
# Needs (same folder as your notebook): lgb_matcher.txt, threshold.txt, student_resource/...
# and the files written by the final pipeline: output/candidate_pairs.tsv, output/matching_results.tsv
# ==========================================

BASE = './student_resource/student_resource/'

# ---------- text cleaning (same as the notebook) ----------
PUNCT = regex.compile(r'[^\p{L}\p{M}\p{N}\s]+')
LATIN_MARKS = regex.compile(r'(?<=\p{Latin})\p{M}+')
NONLATIN = regex.compile(r'[\p{L}--\p{Latin}]', flags=regex.V1)

ADDR_WORDS = {
    "road": "rd", "roads": "rd", "street": "st", "avenue": "ave", "boulevard": "blvd",
    "lane": "ln", "drive": "dr", "highway": "hwy", "parkway": "pkwy",
    "apartment": "apt", "building": "bldg", "suite": "ste", "floor": "fl", "unit": "unit",
    "north": "n", "south": "s", "east": "e", "west": "w",
    "northeast": "ne", "northwest": "nw", "southeast": "se", "southwest": "sw",
    "sector": "sec", "district": "dist", "junction": "jn", "near": "nr",
    "opposite": "opp", "market": "mkt", "number": "no",
}
LEGAL = {
    "pvt", "private", "ltd", "limited", "llp", "opc", "inc", "incorporated", "corp",
    "corporation", "co", "company", "llc", "lp", "plc", "gmbh", "ag", "sa", "sarl",
    "sas", "bv", "nv", "pte", "pty",
}


def clean(text):
    t = unicodedata.normalize('NFD', str(text))
    t = LATIN_MARKS.sub('', t)
    t = unicodedata.normalize('NFKC', unicodedata.normalize('NFC', t)).lower()
    t = t.replace("&", " and ")
    t = re.sub(r"#\s*(\d+)", r"\1", t)
    t = PUNCT.sub(" ", t)
    return re.sub(r"\s+", " ", t).strip()


def norm_name(text):
    return " ".join(w for w in clean(text).split() if w not in LEGAL)


def norm_addr(text):
    return " ".join(ADDR_WORDS.get(w, w) for w in clean(text).split())


def nums(t):
    return {x.lstrip('0') for x in re.findall(r'\d+', t)}


# ---------- features (same columns as training) ----------
def pair_features(n1, a1, n2, a2, cnt_id1, cnt_id2, names):
    f = {}
    f['name_ratio'] = [fuzz.ratio(x, y) for x, y in zip(n1, n2)]
    f['name_tsort'] = [fuzz.token_sort_ratio(x, y) for x, y in zip(n1, n2)]
    f['name_tset'] = [fuzz.token_set_ratio(x, y) for x, y in zip(n1, n2)]
    f['name_partial'] = [fuzz.partial_ratio(x, y) for x, y in zip(n1, n2)]
    f['addr_ratio'] = [fuzz.ratio(x, y) for x, y in zip(a1, a2)]
    f['addr_tsort'] = [fuzz.token_sort_ratio(x, y) for x, y in zip(a1, a2)]
    f['addr_tset'] = [fuzz.token_set_ratio(x, y) for x, y in zip(a1, a2)]
    f['name_exact'] = [int(x == y and x != '') for x, y in zip(n1, n2)]
    f['addr_exact'] = [int(x == y and x != '') for x, y in zip(a1, a2)]
    f['name_first_eq'] = [int(x.split()[:1] == y.split()[:1] and x != '') for x, y in zip(n1, n2)]
    sa = [nums(x) for x in a1]
    sb = [nums(x) for x in a2]
    inter = np.array([len(x & y) for x, y in zip(sa, sb)])
    union = np.array([len(x | y) for x, y in zip(sa, sb)])
    f['num_inter'] = inter
    f['num_jacc'] = np.where(union > 0, inter / np.maximum(union, 1), 0)
    f['num_conflict'] = [int(len(x) > 0 and len(y) > 0 and len(x & y) == 0) for x, y in zip(sa, sb)]
    f['addr1_empty'] = [int(x == '') for x in a1]
    f['addr2_empty'] = [int(x == '') for x in a2]
    nl1 = np.array([int(bool(NONLATIN.search(x))) for x in n1])
    nl2 = np.array([int(bool(NONLATIN.search(x))) for x in n2])
    f['name1_nonlatin'] = nl1
    f['name2_nonlatin'] = nl2
    f['script_mismatch'] = (nl1 != nl2).astype(int)
    f['name_len_diff'] = [abs(len(x) - len(y)) for x, y in zip(n1, n2)]
    f['addr_len_diff'] = [abs(len(x) - len(y)) for x, y in zip(a1, a2)]
    f['name_tok_diff'] = [abs(len(x.split()) - len(y.split())) for x, y in zip(n1, n2)]
    f['cnt_id1'] = [cnt_id1] * len(n1)
    f['cnt_id2'] = [cnt_id2] * len(n1)
    return pd.DataFrame(f)[names].astype('float32')


# ==========================================
# UI
# ==========================================
st.set_page_config(page_title="Entity Matching Explorer", layout="wide")


@st.cache_resource
def load_model():
    return lgb.Booster(model_file='lgb_matcher.txt'), float(open('threshold.txt').read())


@st.cache_data(show_spinner="Loading data (one time, can take 1-3 minutes)...")
def load_demo():
    cols = ['entity_id', 'business_name', 'business_address', 'country']
    kw = dict(sep='\t', dtype=str, keep_default_na=False)

    s1_all = pd.read_csv(BASE + 'dataset/test/test_source1.tsv', usecols=cols, **kw)
    s1_all['c'] = s1_all['country'].str.lower().str.strip()
    # 100 random Source 1 records per country (so France shows up too)
    sample = pd.concat([g.sample(min(100, len(g)), random_state=0) for _, g in s1_all.groupby('c')])
    chosen = set(sample['entity_id'])

    cand = {}
    for ch in pd.read_csv(BASE + 'output/candidate_pairs.tsv', chunksize=200_000, **kw):
        hit = ch[ch['source1_entity_id'].isin(chosen)]
        for a, b in zip(hit['source1_entity_id'], hit['candidate_entity_ids']):
            cand[a] = b.split(',') if b else []

    final = {}
    for ch in pd.read_csv(BASE + 'output/matching_results.tsv', chunksize=200_000, **kw):
        hit = ch[ch['source1_entity_id'].isin(chosen)]
        for a, b in zip(hit['source1_entity_id'], hit['matched_entity_ids']):
            final[a] = b.split(',') if b else []

    need = set(itertools.chain.from_iterable(cand.values()))
    parts, total = [], 0
    for fname in ('test_source2.tsv', 'test_source3.tsv'):
        for ch in pd.read_csv(BASE + 'dataset/test/' + fname, usecols=cols, chunksize=1_000_000, **kw):
            total += len(ch)
            parts.append(ch[ch['entity_id'].isin(need)])
    recs = pd.concat(parts).drop_duplicates('entity_id').set_index('entity_id')
    return sample.set_index('entity_id'), cand, final, recs, total


model, threshold = load_model()
FEATURES = model.feature_name()

st.title("Entity Matching Explorer")
st.caption("Pick a Source 1 business and see which Source 2 / Source 3 records we matched to it, and why.")

tab_explore = st.tabs(["Explore a record"])[0]

# ---------------- Explore ----------------
with tab_explore:
    if st.button("Load data"):
        st.session_state['loaded'] = True
    if not st.session_state.get('loaded'):
        st.info("Click **Load data** to start. It reads your test files and the output files once.")
        st.stop()

    try:
        sample, cand, final, recs, total_s23 = load_demo()
    except Exception as e:
        st.error(f"Could not load the data: {e}")
        st.stop()

    country = st.radio("Country", sorted(sample['c'].unique()), horizontal=True)
    pool = sample[sample['c'] == country]
    sel = st.selectbox(
        "Choose a Source 1 record", pool.index.tolist(),
        format_func=lambda i: f"{pool.loc[i, 'business_name']}   |   {i}")

    row = pool.loc[sel]
    ids = cand.get(sel, [])
    final_ids = set(final.get(sel, []))

    # ----- Step 1 -----
    st.header("1. The Source 1 record")
    n1, a1 = norm_name(row['business_name']), norm_addr(row['business_address'])
    st.dataframe(pd.DataFrame({
        "": ["Name", "Address", "Country"],
        "Original": [row['business_name'], row['business_address'], row['country']],
        "After cleaning": [n1, a1, row['c']],
    }), hide_index=True)
    st.caption("Cleaning: lowercase, remove punctuation, shorten words (road to rd), "
               "drop legal suffixes (pvt, ltd, inc), remove accents, remove zero-padding in numbers.")

    # ----- Step 2 -----
    st.header("2. Blocking: finding candidates")
    st.write(f"Comparing against all **{total_s23:,}** Source 2 and Source 3 records would be far too slow. "
             f"Blocking kept only **{len(ids)}** records that share a key with this business in the same country.")
    if not ids:
        st.warning("No candidates were found for this record, so it has no match.")
        st.stop()

    # ----- Step 3 -----
    st.header("3. Scoring every candidate")
    got = recs.reindex(ids).fillna('')
    X = pair_features(
        [n1] * len(got), [a1] * len(got),
        [norm_name(x) for x in got['business_name']],
        [norm_addr(x) for x in got['business_address']],
        len(ids), 5, FEATURES)   # cnt_id1 is exact; cnt_id2 is a typical value
    got['score'] = model.predict(X)
    got = got.sort_values('score', ascending=False).reset_index()
    got['source'] = got['entity_id'].str[:2]
    got['final'] = np.where(got['entity_id'].isin(final_ids), "MATCH", "")

    c1, c2, c3 = st.columns(3)
    c1.metric("Candidates scored", len(got))
    c2.metric(f"Score at or above {threshold:.3f}", int((got['score'] >= threshold).sum()))
    c3.metric("Final matches", len(final_ids))

    cfg = {"score": st.column_config.ProgressColumn("Score", min_value=0.0, max_value=1.0, format="%.3f")}
    show = ['entity_id', 'source', 'business_name', 'business_address', 'country', 'score', 'final']

    # ----- Step 4 -----
    st.header("4. Final matches")
    matched = got[got['final'] == "MATCH"]
    if len(matched):
        st.dataframe(matched[show], hide_index=True, column_config=cfg)
    else:
        st.write("No candidate passed the decision rule, so this record has no matches.")

    st.subheader("Other candidates that were rejected")
    rest = got[got['final'] != "MATCH"]
    st.dataframe(rest[show].head(15), hide_index=True, column_config=cfg)
    st.caption("Scores here are re-computed in this app. The candidate-pool feature for the Source 2/3 record "
               "is set to a typical value, so a score can differ slightly from the real run. "
               "The MATCH column comes straight from your matching_results.tsv.")

