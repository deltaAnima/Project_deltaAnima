import argparse
import json
import pathlib
import webbrowser

MARKER = 'window.EMBEDDED_CSV = null;'
VAD_MARKER = 'window.EMBEDDED_VAD = null;'


def js_literal(value):
    # Escape "</" so embedded text can't close the <script> tag early.
    return json.dumps(value, ensure_ascii=False).replace('</', '<\\/')


def main():
    parser = argparse.ArgumentParser(description='Open an emotions CSV in viewer.html (data stays on this PC)')
    parser.add_argument('csv', nargs='?', default='emotions.csv')
    parser.add_argument('--output', help='Defaults to <csv name>_view.html next to the CSV')
    parser.add_argument('--vad', help='VAD lexicon JSON (defaults to VAD.json next to the CSV or this script)')
    parser.add_argument('--no-open', action='store_true', help="Don't open the browser")
    args = parser.parse_args()

    here = pathlib.Path(__file__).resolve().parent
    template = (here / 'viewer.html').read_text(encoding='utf-8')
    if MARKER not in template or VAD_MARKER not in template:
        raise SystemExit('viewer.html is missing the embed markers')
    src = pathlib.Path(args.csv)
    text = src.read_text(encoding='utf-8-sig')
    html = template.replace(MARKER, f'window.EMBEDDED_CSV = {js_literal(text)}; window.EMBEDDED_NAME = {js_literal(src.name)};', 1)

    vad = pathlib.Path(args.vad) if args.vad else next(
        (p for p in (src.with_name('VAD.json'), here / 'VAD.json') if p.exists()), None)
    if vad:
        lexicon = json.loads(vad.read_text(encoding='utf-8-sig'))
        html = html.replace(VAD_MARKER, f'window.EMBEDDED_VAD = {js_literal(lexicon)}; window.EMBEDDED_VAD_NAME = {js_literal(vad.name)};', 1)
        print(f'Embedded VAD lexicon {vad} ({len(lexicon)} terms)')
    else:
        print('No VAD.json found; the 3D cube will show the trajectory only')

    out = pathlib.Path(args.output) if args.output else src.with_name(src.stem + '_view.html')
    out.write_text(html, encoding='utf-8')
    print(f'Wrote {out}')
    if not args.no_open:
        webbrowser.open(out.resolve().as_uri())


if __name__ == '__main__':
    main()
