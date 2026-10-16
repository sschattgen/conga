''' Matplotlib versions of the logo-drawing primitives in tcrdist_svg_basic.py
(and conga/svg_basic.py), so that logos can be drawn straight onto a
matplotlib Axes instead of going SVG file -> external converter -> PNG -> imread.

Each primitive here has the same signature as its SVG counterpart and uses the
same SVG-style coordinate system (units of SVG "pixels", y increasing
downward), but returns a small drawing-command tuple instead of an SVG string.
That way the layout code (e.g. make_tcr_logo.make_tcr_logo) can be shared
verbatim between the two renderers by passing in `drawer=mpl_logo_basic`
instead of `drawer=tcrdist_svg_basic`. Use draw_cmds to render the commands.

Glyph geometry reproduces the SVG text_in_box convention exactly: a font-size
100 monospace character is taken to be 60 units wide and 75 units tall
(all_text_width/all_text_height in tcrdist_svg_basic), its baseline sits on
the bottom edge of the box, and it is scaled independently in x and y to fill
the box. Text is drawn as filled glyph outlines (TextPath) from the DejaVu Sans
Mono font that ships with matplotlib, so the output does not depend on system
fonts. Rectangle strokes are drawn as filled outlines in data units (as in SVG)
rather than as matplotlib lines, so they scale with the logo.
'''
from collections import namedtuple
from functools import lru_cache

import numpy as np
from matplotlib.font_manager import FontProperties, findfont
from matplotlib.patches import PathPatch
from matplotlib.path import Path
from matplotlib.textpath import TextPath

from .basic import MONOSPACE_FONT_FAMILY
from .tcrdist_svg_basic import (all_text_height, all_text_width,
                                protein_logo_boxes, stack_boxes)

# drawing commands, in SVG coordinates (y increases downward)
TextInBox = namedtuple('TextInBox', 'upper_left lower_right text color')
Text = namedtuple('Text', 'text lower_left fontsize font_family color font_weight')
Rect = namedtuple('Rect', 'upper_left lower_right fill stroke stroke_width')

# width and height of a monospace character, in units of the font size,
# as assumed by the SVG text_in_box code (which uses font-size 100)
_CHAR_WIDTH = all_text_width[100] / 100.
_CHAR_HEIGHT = all_text_height[100] / 100.


def _font_properties(font_family, font_weight):
    assert font_weight in ['normal', 'bold']
    prop = FontProperties(family=font_family, weight=font_weight)
    # fail loudly instead of silently substituting a different font
    findfont(prop, fallback_to_default=False)
    return prop

_font_properties(MONOSPACE_FONT_FAMILY, 'normal') # check at import time


@lru_cache(maxsize=None)
def _glyph_path(text, font_family, font_weight):
    ''' Returns (vertices, codes) for text at font size 1, baseline at y=0,
    y increasing upward
    '''
    path = TextPath((0, 0), text, size=1,
                    prop=_font_properties(font_family, font_weight),
                    usetex=False)
    return np.array(path.vertices), np.array(path.codes)


## the primitives, same signatures as in tcrdist_svg_basic #####################

def text_in_box( upper_left, lower_right, text, color ):
    return TextInBox(tuple(upper_left), tuple(lower_right), text, color)


def make_text( text, lower_left, fontsize, extra_tag = None,
               font_family = MONOSPACE_FONT_FAMILY, color = "black",
               font_weight = "normal" ):
    assert font_weight in ['normal','bold']
    return Text(text, tuple(lower_left), fontsize, font_family, color,
                font_weight)


def rectangle( upper_left, lower_right, fill, stroke, stroke_width=1,
               dashed=False ):
    assert not dashed # not implemented here, no current logo uses it
    return Rect(tuple(upper_left), tuple(lower_right), fill, stroke,
                stroke_width)


def protein_logo( upper_left, lower_right, pwm, scale={} ):
    return [text_in_box(p0, p1, aa, color) for p0, p1, aa, color in
            protein_logo_boxes(upper_left, lower_right, pwm, scale)]


def make_stack( upper_left, lower_right, l ):
    return [text_in_box(p0, p1, word, color) for p0, p1, word, color in
            stack_boxes(upper_left, lower_right, l)]


## rendering ###################################################################

def _flatten(cmds):
    for cmd in cmds:
        if isinstance(cmd, (TextInBox, Text, Rect)):
            yield cmd
        else:
            assert isinstance(cmd, list), f'unrecognized drawing command: {cmd!r}'
            yield from _flatten(cmd)


def _rect_vertices(x0, y0, x1, y1, reverse=False):
    verts = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
    if reverse:
        verts.reverse()
    return verts + [verts[0]]


def _closed_path(*vertex_lists):
    verts, codes = [], []
    for vl in vertex_lists:
        verts.extend(vl)
        codes.extend([Path.MOVETO] + [Path.LINETO]*(len(vl)-2) + [Path.CLOSEPOLY])
    return Path(verts, codes)


def _patches_for_cmd(cmd):
    ''' Returns a list of (path, color) in SVG coordinates
    '''
    if isinstance(cmd, TextInBox):
        (x0, y0), (x1, y1) = cmd.upper_left, cmd.lower_right
        # same scaling as the SVG version: transform=scale(x_scale,y_scale)
        # applied to a font-size-100 text element with baseline at y1
        x_scale = (x1-x0) / (_CHAR_WIDTH * len(cmd.text))
        y_scale = (y1-y0) / _CHAR_HEIGHT
        verts, codes = _glyph_path(cmd.text, MONOSPACE_FONT_FAMILY, 'normal')
        new_verts = np.column_stack([x0 + x_scale * verts[:,0],
                                     y1 - y_scale * verts[:,1]])
        return [(Path(new_verts, codes), cmd.color)]

    elif isinstance(cmd, Text):
        x0, y0 = cmd.lower_left
        verts, codes = _glyph_path(cmd.text, cmd.font_family, cmd.font_weight)
        new_verts = np.column_stack([x0 + cmd.fontsize * verts[:,0],
                                     y0 - cmd.fontsize * verts[:,1]])
        return [(Path(new_verts, codes), cmd.color)]

    else:
        assert isinstance(cmd, Rect)
        (x0, y0), (x1, y1) = cmd.upper_left, cmd.lower_right
        x0, x1 = min(x0, x1), max(x0, x1)
        y0, y1 = min(y0, y1), max(y0, y1)
        patches = []
        if cmd.fill != 'none':
            patches.append((_closed_path(_rect_vertices(x0, y0, x1, y1)),
                            cmd.fill))
        w = cmd.stroke_width
        if cmd.stroke != 'none' and w > 0:
            # SVG strokes are centered on the edge
            outer = _rect_vertices(x0-w/2, y0-w/2, x1+w/2, y1+w/2)
            if x1-x0 > w and y1-y0 > w: # a ring, with a hole
                inner = _rect_vertices(x0+w/2, y0+w/2, x1-w/2, y1-w/2,
                                       reverse=True)
                patches.append((_closed_path(outer, inner), cmd.stroke))
            else: # stroke covers the whole interior
                patches.append((_closed_path(outer), cmd.stroke))
        return patches


def draw_cmds( ax, cmds, width, height ):
    ''' Draw the commands onto ax, set up to show the SVG canvas
    [0,width] x [0,height] (y increasing downward) with equal aspect ratio,
    the way plt.imshow showed the converted PNG.

    Commands are drawn in order, so later ones are on top, as in SVG.
    '''
    assert width > 0 and height > 0
    for cmd in _flatten(cmds):
        for path, color in _patches_for_cmd(cmd):
            ax.add_patch(PathPatch(path, facecolor=color, edgecolor='none',
                                   linewidth=0))
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)
    ax.set_aspect('equal')
