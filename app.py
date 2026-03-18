import streamlit as st
import filter as filt
import labeller as lab
import plot_data as plot
import pandas as pd
from pathlib import Path
import csv
import base64

BASE_DIR   = Path(__file__).parent          # folder containing this .py file

def resolve_abs(rel_or_abs: str) -> str:
    p = Path(rel_or_abs)
    return str(p if p.is_absolute() else (BASE_DIR / p).resolve())

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".svg"}

st.markdown("""<style>
:root{
  --hover-max-w: 92vw;          /* max overlay width */
  --hover-max-h: 85vh;          /* max overlay height */
  --hover-font-size: 15px;
  --hover-line-height: 1.45;
  --hover-cell-pad: 10px 12px;  /* table cell padding */
}

/* Wrapper around each filename + hover card */
.img-hover{
  position:relative;
  display:inline-block;
  cursor:default;
}

/* Overlay preview (default HIDDEN, shrink-to-fit) */
.img-hover .img-preview{
  display:none;                 /* hidden until hover */
  position:fixed;               /* center on viewport */
  left:50%; top:50%;
  transform:translate(-50%,-50%);
  z-index:1000;

  padding:8px 10px;
  background:#fff;              /* solid background for dark mode */
  border:1px solid #e5e5e5;
  border-radius:10px;
  box-shadow:0 8px 28px rgba(0,0,0,.24);

  /* shrink to content, but never exceed screen bounds */
  width:auto; height:auto;
  max-width:var(--hover-max-w);
  max-height:var(--hover-max-h);
  overflow:auto;                /* scroll if content is tall */
}

/* Show overlay ONLY on hover; center contents neatly */
.img-hover:hover .img-preview{
  display:inline-grid;          /* apply layout only when visible */
  place-items:center;
}

/* General text inside overlay */
.img-hover .img-preview,
.img-hover .img-preview :is(p, span, code, li){
  color:#111 !important;
  font-size:var(--hover-font-size) !important;
  line-height:var(--hover-line-height) !important;
  background-color:transparent;
}

/* ---------- Image previews ---------- */
.img-hover .img-preview img{
  display:block;
  width:auto; height:auto;
  max-width:calc(var(--hover-max-w) - 2rem);
  max-height:calc(var(--hover-max-h) - 2rem);
  object-fit:contain;           /* no crop, no distortion */
  margin:0 auto;                /* centered */
}

/* ---------- CSV table previews ---------- */
.img-hover .img-preview .tbl-wrap{
  max-width:calc(var(--hover-max-w) - 2rem);
  max-height:calc(var(--hover-max-h) - 2rem);
  overflow:auto;
}

.img-hover .img-preview table{
  border-collapse:collapse;
  background:#fff;              /* force white canvas for dark mode */
  table-layout:auto;
}

.img-hover .img-preview th,
.img-hover .img-preview td{
  background:#fff;              /* prevent black-on-black in dark themes */
  border:1px solid #eee;
  padding:var(--hover-cell-pad);
  text-align:left;
  vertical-align:top;
  white-space:normal;
  word-break:break-word;
  color:#111 !important;
}

.img-hover .img-preview th{
  background:#f7f7f7;
  font-weight:600;
}

/* Keep filename chips tidy in the list */
.img-hover code{
  white-space:nowrap;
  line-height:1.2;
}

</style>"""
, unsafe_allow_html=True)

import statistics as stats  # add this

def _split_outside_quotes(line: str, delim: str):
    out, buf, in_q = [], [], False
    for ch in line:
        if ch == '"':
            in_q = not in_q
            buf.append(ch)
        elif ch == delim and not in_q:
            out.append(''.join(buf)); buf = []
        else:
            buf.append(ch)
    out.append(''.join(buf))
    return out

def _score_delim(lines, delim):
    counts = []
    for s in lines:
        if not s: 
            continue
        parts = _split_outside_quotes(s, delim)
        counts.append(max(0, len(parts) - 1))
    if not counts:
        return (0.0, 0.0, 0.0)
    prop  = sum(c > 0 for c in counts) / len(counts)   # lines containing the delim
    med   = float(stats.median(counts))
    stdev = float(stats.pstdev(counts))
    return (prop, stdev, med)

def _read_sample_lines(path, max_lines=200):
    lines = []
    with open(path, encoding="utf-8-sig", errors="ignore") as f:
        for i, line in enumerate(f):
            if i >= max_lines: break
            s = line.rstrip("\r\n")
            if s: lines.append(s)
    return lines

def _pick_delimiter(sample_lines, candidates=(',', ';', '\t', '|')):
    # Choose a delimiter only if it appears on MOST lines and with a stable count
    best = (None, -1.0)
    for d in candidates:
        prop, stdev, med = _score_delim(sample_lines, d)
        # Require ~80% of lines to contain the delim, at least one per line on median,
        # and low variance so commas in sentences don't fool us.
        if prop >= 0.8 and med >= 1 and stdev <= 0.5:
            score = prop - 0.1*stdev + 0.05*med
            if score > best[1]:
                best = (d, score)
    return best[0]

def _likely_header(fields):
    # Basic heuristics: short, mostly alphabetic, common header tokens
    tokens = {'response','text','comment','feedback',
              'label','category','predicted_category','prediction','class'}
    f = [str(x).strip().strip('"').lower() for x in fields]
    if any(x in tokens for x in f): 
        return True
    # If most cells contain letters and aren't too long, treat as header
    alpha = sum(any(c.isalpha() for c in x) for x in f)
    return (alpha / max(1, len(f))) >= 0.6 and all(len(x) <= 64 for x in f)

def _read_single_column_lines(path, max_rows):
    lines = []
    with open(path, encoding="utf-8-sig", errors="ignore") as f:
        for i, line in enumerate(f):
            if i >= max_rows: break
            s = line.rstrip("\r\n")
            if s: lines.append(s)
    return pd.DataFrame({"response": lines})

def render_name_with_hover(display_name: str, abs_path: str, *, max_image_kb: int = 8192, max_rows: int = 10) -> None:
    ext = Path(display_name).suffix.lower()

    # Images → show the image
    if ext in IMAGE_EXTS:
        try:
            data = Path(abs_path).read_bytes()
            if len(data) > max_image_kb * 1024:
                st.markdown(f"`{display_name}`"); return
            mime = "image/svg+xml" if ext == ".svg" else f"image/{'jpeg' if ext in {'.jpg','.jpeg'} else ext.lstrip('.')}"
            b64 = base64.b64encode(data).decode("utf-8")
            st.markdown(
                f"""
                <span class="img-hover">
                  <code>{display_name}</code>
                  <span class="img-preview">
                    <img src="data:{mime};base64,{b64}">
                  </span>
                </span>
                """,
                unsafe_allow_html=True,
            )
        except Exception:
            st.markdown(f"`{display_name}`")
        return

    # CSVs → show first N rows as a tiny table

    if ext == ".csv":
        try:
            # Sample a few lines and decide if there is a trustworthy delimiter
            sample = _read_sample_lines(abs_path, max_lines=200)
            delim = _pick_delimiter(sample)  # returns ',', ';', '\t', '|', or None

            if not delim:
                # No trustworthy delimiter -> treat as a true single-column file
                df = _read_single_column_lines(abs_path, max_rows)  # returns {"response": ...}
            else:
                # Read raw with no header first (we'll decide header ourselves)
                df = pd.read_csv(
                    abs_path,
                    nrows=max_rows,
                    encoding="utf-8-sig",
                    sep=delim,
                    engine="python",
                    quoting=csv.QUOTE_MINIMAL,
                    on_bad_lines="skip",
                    header=None,
                )

                # If the first row looks like a header, promote it
                if df.shape[0] > 0 and _likely_header(df.iloc[0].tolist()):
                    header_vals = [str(x).strip() or f"col{i+1}" for i, x in enumerate(df.iloc[0].tolist())]
                    df = df.iloc[1:].reset_index(drop=True)
                    df.columns = header_vals[:df.shape[1]]
                else:
                    # No header detected: assign friendly names
                    cols = ["response", "predicted_category"] + [f"col{i+3}" for i in range(max(0, df.shape[1]-2))]
                    df.columns = cols[:df.shape[1]]

                # For preview, show at most the first two columns
                if df.shape[1] >= 2:
                    df = df.iloc[:, :2]

                # If after all that we somehow ended with 1 column, normalize name
                if df.shape[1] == 1 and df.columns[0] != "response":
                    df.columns = ["response"]

            # Render compact HTML table inside the hover preview
            table_html = df.to_html(index=False, border=0, escape=True, classes="hover-table")

            st.markdown(
                f"""
                <span class="img-hover">
                <code>{display_name}</code>
                <span class="img-preview">
                    <div class="tbl-wrap">{table_html}</div>
                </span>
                </span>
                """,
                unsafe_allow_html=True,
            )
        except Exception:
            # On any error, fall back to just the filename
            st.markdown(f"`{display_name}`")
        return

    # Everything else → just the name
    st.markdown(f"`{display_name}`")

st.set_page_config(layout="wide") 
st.title("Survey Response Processor")

# --- state ---
MAX_BOXES = 5
if "file_count" not in st.session_state:
    st.session_state.file_count = 1
if "processed" not in st.session_state:
    st.session_state.processed = [None] * 5
if "filtered_names" not in st.session_state:
    st.session_state.filtered_names = [None] * MAX_BOXES  # list[str] per column
if "labelled_names" not in st.session_state:
    st.session_state.labelled_names = [None] * MAX_BOXES  # list[str] per column
if "pie_charts" not in st.session_state:
    st.session_state.pie_charts = [None] * MAX_BOXES  # list[str] per column
if "preview_paths" not in st.session_state:
    st.session_state.preview_paths = {}

def add_box():
    st.session_state.file_count = min(st.session_state.file_count + 1, 5)

def remove_box():
    if st.session_state.file_count > 1:
        idx = st.session_state.file_count - 1  # index being removed
        st.session_state.pop(f"file_{idx}", None)
        st.session_state.pop(f"period_{idx}", None)
        st.session_state.processed[idx] = None
        st.session_state.filtered_names[idx] = None
        st.session_state.labelled_names[idx] = None
        st.session_state.pie_charts[idx] = None
        st.session_state.file_count -= 1

# --- controls row ---
c1, c2, c3 = st.columns([8, 1, 1])
with c1:
    st.caption(f"All input CSVs should have only 1 column of responses with no header, and be in the Unlabelled folder ( {st.session_state.file_count} / 5 )")
with c2:
    st.button("➕", on_click=add_box, use_container_width=True,
              disabled=st.session_state.file_count >= 5)
with c3:
    st.button("➖", on_click=remove_box, use_container_width=True,
              disabled=st.session_state.file_count <= 1)

c_label_in, *cols_in = st.columns([0.8, 1, 1, 1, 1, 1], gap="small")
example_dates = ["jan_25", "feb_25", "march_25", "april_25", "may_25"]
error_slots = [None]*5
with c_label_in:
    st.markdown("**Inputs**")  # or st.caption(" ") to keep height consistent

for i, col in enumerate(cols_in[:st.session_state.file_count]):
    with col:
        st.text_input(f"File {i+1}", key=f"file_{i}", placeholder=f"responses_{i+1}.csv")
        st.text_input(f"Time Period {i+1}", key=f"period_{i}", placeholder=example_dates[i])
        error_slots[i] = st.empty()


if st.button("Submit"):
    with st.spinner("Processing files…"):

        to_run = []                 # [(i, file, period), ...]
        index_by_period = {}        # {period: i}  (useful if your filter returns a dict keyed by period)
        problems = 0

        # validate + gather
        for i in range(st.session_state.file_count):
            file = (st.session_state.get(f"file_{i}", "") or "").strip()
            period = (st.session_state.get(f"period_{i}", "") or "").strip()

            if file and not period:
                error_slots[i].error("Please enter a time period for this file.")
                problems += 1
                st.session_state.processed[i] = None
                continue
            if period and not file:
                error_slots[i].error("Please enter a file name for this time period.")
                problems += 1
                st.session_state.processed[i] = None
                continue
            if file and period:
                to_run.append((i, file, period))
                index_by_period[period] = i
            else:
                st.session_state.processed[i] = None  # keep row tidy

        if problems:
            st.error(f"❗Please fix the {problems} highlighted item(s) above.")
            st.stop()
        if not to_run:
            st.warning("Nothing to submit. Add at least one file and a time period.")
            st.stop()

        entries_dict = {period: file for (i, file, period) in to_run}
        batch = filt.format_batch_for_filtering(entries_dict)
        filter_batch_info = filt.filter_batch(batch)

        for period, files in filter_batch_info.items():
            i = index_by_period.get(period)
            if i is None:
                continue

            filtered = files["filtered"][0] + f"  ({files['filtered'][1]})"
            st.session_state.processed[i] = filtered
            st.session_state.filtered_names[i] = files["filtered"][0]

        st.subheader("Results")

        c_label1, *cols1 = st.columns([0.8, 1, 1, 1, 1, 1], gap="small")
        with c_label1:
            st.markdown("**Filtered Files:**")

        for i, col in enumerate(cols1[:st.session_state.file_count]):
            with col:
                items = st.session_state.filtered_names[i]  # can be str or list[str] (relative or absolute)
                if not items:
                    st.markdown("—")
                else:
                    paths = items if isinstance(items, list) else [items]
                    for rel_or_abs in paths:
                        abs_path = resolve_abs(rel_or_abs)         # make absolute
                        name = Path(abs_path).name                 # nice display name
                        render_name_with_hover(name, abs_path)     # hover shows CSV preview / image

        st.divider()

        labeller_input = {}
        for i in range(st.session_state.file_count):
            period = (st.session_state.get(f"period_{i}", "") or "").strip()
            names  = st.session_state.filtered_names[i] or []
            if period and names:
                labeller_input[period] = names

        labelled_by_period = lab.label_batch(labeller_input)  # {period: [labelled.csv, ...]}

        for i in range(st.session_state.file_count):
            st.session_state.labelled_names[i] = []

        # Put each period's labelled names under its matching column
        for period, filename in labelled_by_period.items():
            i = index_by_period.get(period)
            if i is not None:
                st.session_state.labelled_names[i] = filename
        
        c_label2, *cols2 = st.columns([0.8, 1, 1, 1, 1, 1], gap="small")
        with c_label2:
            st.markdown("**Labelled Files:**")

        for i, col in enumerate(cols2[:st.session_state.file_count]):
            with col:
                items = st.session_state.labelled_names[i]  # can be str or list[str] (relative or absolute)
                if not items:
                    st.markdown("—")
                else:
                    paths = items if isinstance(items, list) else [items]
                    for rel_or_abs in paths:
                        abs_path = resolve_abs(rel_or_abs)         # make absolute
                        name = Path(abs_path).name                 # nice display name
                        render_name_with_hover(name, abs_path)     # hover shows CSV preview / image

        st.divider()

        plot_input = {}
        for i in range(st.session_state.file_count):
            period = (st.session_state.get(f"period_{i}", "") or "").strip()
            names  = st.session_state.labelled_names[i] or []
            if period and names:
                plot_input[period] = names
        print(plot_input)
        plot_output = plot.plot_batch(plot_input)

        print(plot_output)

        for i in range(st.session_state.file_count):
            st.session_state.pie_charts[i] = []

        # Put each period's labelled names under its matching column
        for period, filename in plot_output["pie"].items():
            i = index_by_period.get(period)
            if i is not None:
                st.session_state.pie_charts[i] = filename
        
        c_label3, *cols3 = st.columns([0.8, 1, 1, 1, 1, 1], gap="small")
        with c_label3:
            st.markdown("**Pie Charts:**")
        for i, col in enumerate(cols3[:st.session_state.file_count]):
            with col:
                item = st.session_state.pie_charts[i]   # can be str or list[str] (relative paths)
                if not item:
                    st.markdown("—")
                else:
                    paths = item if isinstance(item, list) else [item]
                    for rel in paths:
                        abs_path = resolve_abs(rel)
                        name = Path(abs_path).name
                        render_name_with_hover(name, abs_path)

        st.divider()

        filecount = st.session_state.file_count

        c_label4, *cols4 = st.columns([0.8, filecount, 5 - filecount], gap="small")
        with c_label4:
            st.markdown("**Heatmap:**")
        with cols4[0]:
            item = plot_output.get("heatmap")
            if not item:
                st.markdown("—")
            else:
                paths = item if isinstance(item, list) else [item]
                for rel in paths:
                    abs_path = resolve_abs(rel)
                    name = Path(abs_path).name
                    render_name_with_hover(name, abs_path)

        st.divider()

        c_label5, *cols5 = st.columns([0.8, filecount, 5 - filecount], gap="small")
        with c_label5:
            st.markdown("**Table:**")
        with cols5[0]:
            item = plot_output.get("table")
            if not item:
                st.markdown("—")
            else:
                paths = item if isinstance(item, list) else [item]
                for rel in paths:
                    abs_path = resolve_abs(rel)
                    name = Path(abs_path).name
                    render_name_with_hover(name, abs_path)


    st.success("Done!")