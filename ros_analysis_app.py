"""Streamlit gallery for batch ROS intensity / DAPI nucleus quantification."""
from __future__ import annotations
import re
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import streamlit as st
from matplotlib import pyplot as plt
from skimage import exposure
from ros_analysis_core import analyze_pair, csv_bytes, pair_key, read_tiff

@dataclass
class ImageRecord:
    folder: str
    stem: str
    ros_name: str
    dapi_name: str
    ros: np.ndarray
    dapi: np.ndarray
    ros_integrated: float
    ros_background: float
    nuclei_count: int
    nuclei_labels: np.ndarray
    per_cell: float
    warnings: list[str]

def main():
    st.set_page_config(page_title="ROS Fluorescence Analysis", page_icon="🧫", layout="wide")
    st.markdown("""<style>
    .stApp{background:#f5f7fb;color:#182438}.block-container{max-width:1320px;padding-top:2rem}
    [data-testid="stSidebar"]{background:#f0f4f8;border-right:1px solid #dce4ed}
    [data-testid="stSidebar"] *{color:#182438}
    [data-testid="stSidebar"] [data-baseweb="select"] *,[data-testid="stSidebar"] input{color:#182438!important}
    [data-testid="stSidebar"] [data-baseweb="slider"] [role="slider"]{background:#316b91}
    .hero{padding:1.5rem 1.8rem;border-radius:18px;background:linear-gradient(120deg,#10243e,#244d70);color:white;margin-bottom:1.2rem}
    .hero h1{color:white;margin:0}.hero p{color:#d3e2f0;margin:.4rem 0 0}
    [data-testid="stMetric"]{background:white;border:1px solid #e3e9f1;padding:1rem;border-radius:14px}
    [data-testid="stMetric"]{padding:.55rem .6rem;min-width:0;overflow:visible}
    [data-testid="stMetricLabel"]{font-size:.72rem!important;white-space:normal!important}
    [data-testid="stMetricValue"]{font-size:clamp(.85rem,1.3vw,1.15rem)!important;white-space:nowrap!important;letter-spacing:-.025em}
    </style>""", unsafe_allow_html=True)
    st.markdown('<div class="hero"><h1>ROS fluorescence analysis</h1><p>Batch quantify ROS signal and normalize by DAPI-positive nuclei.</p></div>', unsafe_allow_html=True)
    with st.sidebar:
        st.header("Analysis settings")
        st.caption("Tune on representative images, then inspect the QC overlays.")
        plane = st.number_input("Z / plane index", min_value=0, value=0, step=1)
        correction = st.selectbox("ROS background correction", ["None", "Subtract estimated background"])
        percentile = st.slider("Background percentile", 0.0, 50.0, 10.0, 0.5, disabled=correction == "None", help="Low percentile used as a uniform per-pixel background estimate.")
        sigma = st.slider("DAPI smoothing (pixels)", 0.0, 5.0, 1.0, 0.25)
        multiplier = st.slider("DAPI Otsu threshold multiplier", 0.3, 2.0, 1.0, 0.05)
        min_area = st.number_input("Minimum DAPI object area (pixels)", min_value=1, value=150, step=25,
                                   help="Objects smaller than this are removed as specks. Set this based on the smallest real nucleus in your images.")
        max_area = st.number_input("Maximum nucleus area (0 = no limit)", min_value=0, value=0, step=50)
        split = st.checkbox("Split touching DAPI objects", value=False, help="Off counts each connected DAPI-positive object once. Turn on only when touching nuclei need to be separated.")
    st.subheader("1 · Choose the image folder")
    local_path = st.text_input("Folder path on this computer / server", placeholder=r"C:\data\experiment_01", help="Use this when the app can access the folder directly.")
    uploads = st.file_uploader("Or upload a folder of TIFF files", type=["tif", "tiff"], accept_multiple_files="directory", help="Folder upload keeps relative subfolder paths for pairing.")
    entries=[]
    if uploads: entries=[(f.name, f) for f in uploads]
    elif local_path.strip():
        root=Path(local_path.strip()).expanduser()
        if root.is_dir(): entries=[(p.relative_to(root).as_posix(), p) for p in sorted(root.rglob("*")) if p.is_file() and p.suffix.lower() in (".tif", ".tiff")]
        else: st.error("That folder path does not exist or is not a folder.")
    if not entries:
        st.info("Choose a folder or upload TIFF files to begin. Names must include C002 for ROS and C001 for DAPI.")
        st.stop()
    groups={}
    for name,obj in entries:
        channel="ros" if re.search(r"(?i)C002", Path(name).name) else "dapi" if re.search(r"(?i)C001", Path(name).name) else None
        if channel: groups.setdefault(pair_key(name), {"ros":[],"dapi":[]})[channel].append((name,obj))
    matched=[(key,g) for key,g in sorted(groups.items()) if len(g["ros"])==1 and len(g["dapi"])==1]
    bad=[(key,g) for key,g in groups.items() if len(g["ros"])!=1 or len(g["dapi"])!=1]
    ignored=len(entries)-sum(len(g["ros"])+len(g["dapi"]) for g in groups.values())
    a,b,c=st.columns(3); a.metric("TIFF files",len(entries)); b.metric("Matched pairs",len(matched)); c.metric("Needs attention",len(bad)+ignored)
    if bad or ignored:
        with st.expander("Pairing validation and file issues",expanded=True):
            for key,g in bad: st.warning(f"{key}: expected one C002 and one C001; found {len(g['ros'])} ROS and {len(g['dapi'])} DAPI file(s).")
            if ignored: st.info(f"{ignored} TIFF file(s) without C001 or C002 were ignored.")
    if not matched:
        st.error("No unambiguous ROS/DAPI pairs found. Keep channels in the same folder and use matching filename stems around C001/C002.")
        st.stop()
    run_key = (tuple((n, g["ros"][0][0], g["dapi"][0][0]) for n, g in matched), int(plane), correction,
               percentile, sigma, multiplier, int(min_area), int(max_area), split)
    if st.session_state.get("ros_run_key") != run_key:
        st.session_state.pop("ros_records", None)
        st.session_state.pop("ros_failures", None)
    if st.button("Analyze matched pairs",type="primary",use_container_width=True):
        records=[]; failures=[]; progress=st.progress(0)
        for i,(key,g) in enumerate(matched):
            rn,ro=g["ros"][0]; dn,do=g["dapi"][0]
            try:
                if hasattr(ro,"seek"): ro.seek(0)
                if hasattr(do,"seek"): do.seek(0)
                rimg=read_tiff(ro,int(plane)); dimg=read_tiff(do,int(plane))
                result=analyze_pair(rimg,dimg,background_percentile=percentile,background_correction=correction!="None",gaussian_sigma=sigma,threshold_multiplier=multiplier,min_area=min_area,max_area=max_area,split_touching=split)
                folder=Path(rn.replace("\\","/")).parent.as_posix()
                if folder==".": folder="(uploaded files)"
                notes=[]
                if result["nuclei_count"]==0: notes.append("No nuclei detected; check DAPI settings")
                if not np.isfinite(result["dapi_threshold"]): notes.append("DAPI image is constant or invalid")
                stem=re.sub(r"(?i)(?:[_ .-]?C002)","",Path(rn).stem).strip("_ .-")
                records.append(ImageRecord(folder,stem,Path(rn).name,Path(dn).name,rimg,dimg,result["ros_integrated"],result["ros_background"],result["nuclei_count"],result["nuclei_labels"],result["ros_per_nucleus"],notes))
            except Exception as exc: failures.append(f"{rn} / {dn}: {exc}")
            progress.progress((i+1)/len(matched))
        st.session_state["ros_records"]=records; st.session_state["ros_failures"]=failures; st.session_state["ros_run_key"]=run_key
    records=st.session_state.get("ros_records",[]); failures=st.session_state.get("ros_failures",[])
    if failures:
        with st.expander(f"{len(failures)} image pair(s) failed",expanded=True):
            for msg in failures: st.error(msg)
    if records:
        st.subheader("2 · Results and quality control")
        st.caption("Integrated ROS is the sum of pixel values. Optional correction subtracts the selected percentile estimate from each pixel. Cell count is estimated from DAPI-positive objects (one nucleus per cell); DAPI alone does not show whole-cell boundaries.")
        st.download_button("Download results CSV",csv_bytes(records),"ros_analysis_results.csv","text/csv",type="primary")
        st.markdown(f"**{len(records)} pair(s) analyzed** · Compact gallery; each card shows both channels and measurements.")
        gallery_columns=st.columns(2)
        for n,r in enumerate(records,1):
            with gallery_columns[(n-1)%2]:
                with st.container(border=True):
                    st.markdown(f"**{n}. {r.folder}**  \n`{r.stem}`")
                    ros_col,dapi_col=st.columns(2)
                    with ros_col:
                        fig,ax=plt.subplots(figsize=(3.4,2.35)); ax.imshow(exposure.rescale_intensity(r.ros,in_range="image"),cmap="gray",aspect="equal")
                        ax.set_axis_off(); fig.tight_layout(pad=0); st.pyplot(fig,use_container_width=True); plt.close(fig)
                        st.caption("ROS")
                    with dapi_col:
                        fig,ax=plt.subplots(figsize=(3.4,2.35)); ax.imshow(exposure.rescale_intensity(r.dapi,in_range="image"),cmap="gray",aspect="equal")
                        ax.contour(r.nuclei_labels>0,levels=[0.5],colors="#30d6b0",linewidths=.55); ax.set_axis_off(); fig.tight_layout(pad=0)
                        st.pyplot(fig,use_container_width=True); plt.close(fig)
                        st.caption("DAPI · counted objects")
                    value_cols=st.columns(3)
                    value_cols[0].metric("Integrated ROS",f"{r.ros_integrated:,.3g}")
                    value_cols[1].metric("Estimated cell count",f"{r.nuclei_count:,}")
                    value_cols[2].metric("ROS per estimated cell",f"{r.per_cell:,.3g}" if np.isfinite(r.per_cell) else "N/A")
                    for note in r.warnings: st.warning(note)
    st.caption("DAPI-based count treats each connected DAPI-positive object as one cell. It is a nucleus-based estimate, not a whole-cell boundary count. Review overlays and tune segmentation on representative images.")

if __name__=="__main__": main()
