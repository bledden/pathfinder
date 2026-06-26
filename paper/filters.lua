-- pandoc Lua filter for the Pathfinder PDF build.
-- Operates at the AST level (robust): real $$..$$ / $x$ math become Math elements and are
-- NEVER touched here, so they compile natively; currency $100 is handled by pandoc's own
-- currency heuristic. We only rewrite literal Unicode that appears in plain text (Str).

local SUP = {['⁰']='0',['¹']='1',['²']='2',['³']='3',['⁴']='4',['⁵']='5',['⁶']='6',
             ['⁷']='7',['⁸']='8',['⁹']='9',['⁺']='+',['⁻']='-'}
-- math glyphs -> emitted as $..$
local MATH = {
  ['μ']='\\mu', ['µ']='\\mu', ['×']='\\times', ['→']='\\to', ['∈']='\\in',
  ['≈']='\\approx', ['≥']='\\ge', ['≤']='\\le', ['±']='\\pm', ['Λ']='\\Lambda',
  ['σ']='\\sigma', ['χ']='\\chi', ['α']='\\alpha', ['Δ']='\\Delta', ['⇒']='\\Rightarrow',
  ['↔']='\\leftrightarrow', ['↓']='\\downarrow', ['⟩']='\\rangle', ['⟨']='\\langle',
  ['≳']='\\gtrsim', ['≪']='\\ll', ['≫']='\\gg', ['√']='\\surd', ['∞']='\\infty',
  ['≅']='\\approx', ['∑']='\\sum', ['⋅']='\\cdot', ['·']='\\cdot',
}
-- text/raw glyphs (no surrounding $)
local TEXT = {
  ['−']='-', ['✓']='\\ding{51}', ['✗']='\\ding{55}', ['★']='\\ding{72}', ['✦']='\\ding{72}',
}
-- everything else non-ASCII (— – § … † ö curly quotes) renders fine in Latin Modern: leave it.

function Str(el)
  local s = el.text
  if not s:match('[\194-\244]') then return nil end   -- pure ASCII fast path
  local out, buf, sup = {}, {}, {}
  local function flushbuf() if #buf>0 then table.insert(out, pandoc.Str(table.concat(buf))); buf={} end end
  local function flushsup() if #sup>0 then table.insert(out, pandoc.RawInline('latex','$^{'..table.concat(sup)..'}$')); sup={} end end
  for _, cp in utf8.codes(s) do
    local ch = utf8.char(cp)
    if SUP[ch] then flushbuf(); table.insert(sup, SUP[ch])
    else
      flushsup()
      if MATH[ch] then flushbuf(); table.insert(out, pandoc.RawInline('latex','$'..MATH[ch]..'$'))
      elseif TEXT[ch] then flushbuf(); table.insert(out, pandoc.RawInline('latex', TEXT[ch]))
      else table.insert(buf, ch) end
    end
  end
  flushsup(); flushbuf()
  return out
end

-- Make inline code (file paths / identifiers) breakable so they don't run off the page or
-- overlap in tables. INLINE Code only, not CodeBlocks.
function Code(el)
  local t = el.text
  t = t:gsub('\\', '\\textbackslash{}')
  t = t:gsub('([{}%%%$#&_])', '\\%1')
  t = t:gsub('~', '\\textasciitilde{}')
  t = t:gsub('%^', '\\textasciicircum{}')
  t = t:gsub('([/_%.%-,])', '%1\\allowbreak{}')
  return pandoc.RawInline('latex', '\\texttt{' .. t .. '}')
end

-- Force consistent figure width (pandoc's \pandocbounded only shrinks-to-fit, never upscales).
function Image(el)
  return pandoc.RawInline('latex',
    '\\includegraphics[width=0.92\\linewidth,keepaspectratio]{' .. el.src .. '}')
end

-- Drop pandoc's auto figure-caption (built from alt text) so it doesn't duplicate the manual
-- "**Figure N.** ..." paragraph the author wrote below each image. Keep the image, centered.
function Figure(fig)
  local out = pandoc.List()
  out:insert(pandoc.RawBlock('latex', '\\begin{center}'))
  out:extend(fig.content)
  out:insert(pandoc.RawBlock('latex', '\\end{center}'))
  return out
end
