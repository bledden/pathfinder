#!/usr/bin/env python3
"""Build a readable PDF/LaTeX draft of pathfinder.md (local, for pre-arXiv review).

Markdown carries lots of literal Unicode (μ, ×, χ², ×10⁻⁷, ≈, ≤, →, ∈, ✓/✗ …) that XeTeX's
default font would render as missing-glyph boxes — and those appear in headline claims. So we
preprocess a TEMP copy, mapping every non-ASCII math glyph to proper LaTeX (routed through math
mode, which always renders) and dingbats to amssymb/pifont, then run pandoc + tectonic.
The real paper/pathfinder.md is never modified.

Usage:  python paper/build_pdf.py            # writes paper/pathfinder_draft.pdf (+ .tex)
"""
import os, re, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, 'pathfinder.md')
TMP = os.path.join(HERE, '_pathfinder_build.md')
HDR = os.path.join(HERE, '_header.tex')
PDF = os.path.join(HERE, 'pathfinder_draft.pdf')
TEX = os.path.join(HERE, 'pathfinder_draft.tex')

PANDOC = '/Users/bledden/.local/bin/pandoc'
TECTONIC = '/Users/bledden/.local/bin/tectonic'

def rawl(latex):
    # pandoc raw-LaTeX inline span: bypasses markdown's $-math parsing (incl. the
    # anti-currency "closing $ followed by digit isn't math" rule that breaks ×10, ≈61, etc.)
    return '`' + latex + '`{=latex}'

SUP = {'⁰':'0','¹':'1','²':'2','³':'3','⁴':'4','⁵':'5','⁶':'6','⁷':'7','⁸':'8','⁹':'9','⁺':'+','⁻':'-'}
# per-glyph -> raw-LaTeX span
MATH = {
    'μ':rawl(r'$\mu$'), 'µ':rawl(r'$\mu$'), '×':rawl(r'$\times$'), '→':rawl(r'$\to$'), '∈':rawl(r'$\in$'),
    '−':'-', '≈':rawl(r'$\approx$'), '≥':rawl(r'$\ge$'), '≤':rawl(r'$\le$'), '±':rawl(r'$\pm$'),
    'Λ':rawl(r'$\Lambda$'), 'σ':rawl(r'$\sigma$'), 'χ':rawl(r'$\chi$'), 'α':rawl(r'$\alpha$'), 'Δ':rawl(r'$\Delta$'),
    '⇒':rawl(r'$\Rightarrow$'), '↔':rawl(r'$\leftrightarrow$'), '↓':rawl(r'$\downarrow$'),
    '⟩':rawl(r'$\rangle$'), '⟨':rawl(r'$\langle$'), '≳':rawl(r'$\gtrsim$'), '≪':rawl(r'$\ll$'), '≫':rawl(r'$\gg$'),
    '√':rawl(r'$\surd$'), '∞':rawl(r'$\infty$'), '≅':rawl(r'$\approx$'),
    '✓':rawl(r'\ding{51}'), '✗':rawl(r'\ding{55}'), '★':rawl(r'$\bigstar$'), '✦':rawl(r'$\bigstar$'),
}

def preprocess(body):
    # No text munging: unicode is handled by the Lua Str filter at the AST level, and real
    # $$..$$ / $x$ math + currency $100 are handled natively by pandoc (tex_math_dollars).
    return body

def main():
    raw = open(SRC, encoding='utf-8').read().split('\n')
    # extract title (first '# ' line) + drop the title/author/affiliation block up to first '---'
    title = raw[0].lstrip('# ').strip()
    try:
        cut = raw.index('---')          # first thematic break ends the title block
    except ValueError:
        cut = 0
    body = '\n'.join(raw[cut+1:])
    body = preprocess(body)
    open(TMP, 'w', encoding='utf-8').write(body)
    open(HDR, 'w', encoding='utf-8').write(
        '\\usepackage{graphicx}\n'  # filters.lua emits raw \includegraphics; ensure graphicx is loaded
        '\\usepackage{amssymb}\n\\usepackage{pifont}\n'
        '\\usepackage{booktabs}\n\\usepackage{etoolbox}\n'
        # body slightly small; wide multi-column tables get scriptsize so they fit the page
        '\\AtBeginDocument{\\small}\n'
        '\\AtBeginEnvironment{longtable}{\\scriptsize}\n'
        '\\setlength{\\tabcolsep}{4pt}\n'
        # wrap long code-block / verbatim lines (appendix repro commands, URLs)
        '\\usepackage{fvextra}\n'
        '\\fvset{breaklines=true,breakanywhere=true,fontsize=\\footnotesize}\n')
    common = [
        '--from', 'markdown+pipe_tables+raw_attribute+raw_tex',
        '--standalone', '--toc', '--toc-depth=3',
        '--lua-filter', os.path.join(HERE, 'filters.lua'),
        '--metadata', f'title={title}',
        '--metadata', 'author=Blake Ledden — Independent Researcher, San Francisco, CA',
        '--metadata', 'date=June 2026',
        '-V', 'geometry:margin=0.9in', '-V', 'fontsize=10pt',
        '-V', 'colorlinks=true', '-V', 'linkcolor=blue', '-V', 'urlcolor=blue',
        '--include-in-header', HDR,
        '--resource-path', HERE + ':' + os.path.dirname(HERE),
    ]
    # 1) emit .tex for inspection (no engine needed)
    subprocess.run([PANDOC, TMP, *common, '-o', TEX], check=True, cwd=HERE)
    print('wrote', TEX)
    # 2) emit PDF via tectonic
    r = subprocess.run([PANDOC, TMP, *common, '--pdf-engine', TECTONIC, '-o', PDF], cwd=HERE)
    if r.returncode == 0:
        sz = os.path.getsize(PDF)
        print(f'wrote {PDF} ({sz//1024} KB)')
    else:
        print('PDF build failed (rc=%d); the .tex is available for manual compile' % r.returncode)
        sys.exit(r.returncode)

if __name__ == '__main__':
    main()
