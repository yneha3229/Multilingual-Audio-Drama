"""Phase 5: the web app.

A small Streamlit page over the pipeline: paste a story, pick languages, and
listen to the episodes.

Run with:
    streamlit run app.py
"""

import hashlib
from pathlib import Path

import streamlit as st

import sarvam_client
from make_drama import run_pipeline
from script_writer import LANGUAGES, NARRATOR

STORIES_DIR = Path("stories")
SAMPLE_STORY = STORIES_DIR / "sample_story.txt"


def folder_name(story: str) -> str:
    """Folder under output/ for this story.

    A story that matches a file in stories/ uses that file's name, so the app
    and the command line share saved work. Anything else gets a name made from
    its text, so pasting the same story twice reuses the first result.
    """
    for path in STORIES_DIR.glob("*.txt"):
        if path.read_text(encoding="utf-8").strip() == story.strip():
            return path.stem
    return "story_" + hashlib.sha1(story.strip().encode("utf-8")).hexdigest()[:10]


def show_cast(result: dict) -> None:
    script = result["script"]
    voices = result["episodes"][0]["manifest"]["voices"]
    rows = [
        {"Character": NARRATOR, "Voice": voices[NARRATOR], "Gender": "", "About": "Tells the story"}
    ]
    rows += [
        {"Character": c.name, "Voice": voices[c.name], "Gender": c.gender, "About": c.description}
        for c in script.characters
    ]
    st.subheader("Cast")
    st.caption("Each character keeps the same voice in every language.")
    st.dataframe(rows, hide_index=True, width="stretch")


def show_quality(episode: dict) -> None:
    report = episode["report"]
    if not report:
        st.info("Quality check not run for this episode. Tick the box in the sidebar to run it.")
        return

    summary = report["summary"]
    one, two, three, four = st.columns(4)
    one.metric("Lines checked", summary["lines"])
    two.metric("Bad takes found", summary["failing_before"])
    three.metric("Still failing", summary["failing_after"])
    four.metric("Character error rate", f"{summary['cer_after']:.1%}")
    st.caption(
        "Every clip is transcribed back to text with speech-to-text and compared with the "
        "script. A clip fails if more than 15% of its characters differ, or if it is spoken "
        "at under 70% of the episode's usual speed."
    )

    by_line = {entry["line"]: entry for entry in episode["manifest"]["lines"]}
    for line in report["lines"]:
        if line["status"] == "ok":
            continue
        clip = episode["folder"] / by_line[line["line"]]["clip"]
        rejected = clip.parent / "rejected" / clip.name
        with st.container(border=True):
            st.markdown(
                f"**Line {line['line']}, {line['speaker']}** was "
                f"{' and '.join(line['problems'])} and is now **{line['status']}**"
            )
            st.write(line["text"])
            before, after = st.columns(2)
            before.caption(f"First take: {line['before']['rate']} characters per second")
            if rejected.exists():
                before.audio(str(rejected))
            after.caption(f"Kept take: {line['after']['rate']} characters per second")
            after.audio(str(clip))

    st.dataframe(
        [
            {
                "Line": line["line"],
                "Speaker": line["speaker"],
                "Result": line["status"],
                "Character error": f"{line['after']['cer']:.1%}",
                "Word error": f"{line['after']['wer']:.1%}",
                "Heard as": line["after"]["transcript"],
            }
            for line in report["lines"]
        ],
        hide_index=True,
        width="stretch",
    )


def show_episode(episode: dict) -> None:
    manifest = episode["manifest"]
    minutes, seconds = divmod(round(manifest["duration"]), 60)
    st.markdown(f"### {episode['script'].title}")
    st.caption(f"{len(manifest['lines'])} lines, {minutes}m {seconds:02d}s")
    st.audio(str(episode["folder"] / manifest["episode"]))
    with_music = episode["folder"] / "episode_with_music.wav"
    if with_music.exists():
        st.caption("With background music (Stable Audio 3)")
        st.audio(str(with_music))

    script_tab, quality_tab = st.tabs(["Script", "Quality check"])
    with script_tab:
        st.dataframe(
            [
                {
                    "Starts at": f"{int(entry['start'] // 60)}:{int(entry['start'] % 60):02d}",
                    "Speaker": entry["speaker"],
                    "Emotion": entry["emotion"],
                    "Line": entry["text"],
                }
                for entry in manifest["lines"]
            ],
            hide_index=True,
            width="stretch",
        )
    with quality_tab:
        show_quality(episode)


st.set_page_config(page_title="Multilingual Audio Drama", page_icon="🎧", layout="wide")
st.title("Multilingual audio drama generator")
st.caption(
    "Paste a short story in English or Hindi. It is split into a script with a narrator and "
    "characters, voiced with Sarvam AI text-to-speech, and translated into other Indian languages."
)

with st.sidebar:
    st.header("Settings")
    extra_languages = st.multiselect(
        "Also translate into",
        options=list(LANGUAGES),
        default=["hi-IN"],
        format_func=lambda code: LANGUAGES[code],
        help="The story's own language is always produced.",
    )
    run_check = st.checkbox(
        "Run the quality check",
        value=True,
        help="Transcribes every clip to find bad takes and re-generates them. "
        "Uses extra Sarvam speech-to-text credits.",
    )
    fresh = st.checkbox(
        "Rewrite the script",
        value=False,
        help="Ignore the saved script for this story and ask the model again.",
    )

default_story = SAMPLE_STORY.read_text(encoding="utf-8") if SAMPLE_STORY.exists() else ""
story = st.text_area(
    "Story",
    value=default_story,
    height=300,
    help="Put dialogue inside double quotation marks so it can be told apart from narration.",
)

if st.button("Generate audio drama", type="primary", disabled=not story.strip()):
    with st.status("Working...", expanded=True) as status:
        try:
            st.session_state["result"] = run_pipeline(
                story,
                folder_name(story),
                extra_languages,
                check=run_check,
                fresh=fresh,
                log=status.write,
            )
            status.update(label="Done", state="complete", expanded=False)
        except (sarvam_client.SarvamError, RuntimeError, ValueError) as error:
            st.session_state.pop("result", None)
            status.update(label="Failed", state="error")
            st.error(str(error))

result = st.session_state.get("result")
if result:
    show_cast(result)
    st.subheader("Episodes")
    tabs = st.tabs([LANGUAGES[episode["code"]] for episode in result["episodes"]])
    for tab, episode in zip(tabs, result["episodes"]):
        with tab:
            show_episode(episode)
