#!/usr/bin/python
# -=- encoding: utf-8 -=-
# Author: Alexandre Bourget 
# Copyright (c) 2008: Alexandre Bourget 
# LICENSE: GPLv3

# How to use this script
# ------------------------
# Create a "content" labeled layer and put a text box (no flowRect), with each
# line looking like:
#
#   background, layer1
#   background, layer2
#   background, layer2, layer3
#   +layer4
#   background, layer2 * 0.5, layer3 * 0.5, layer5
#
# otherwise inkscape's pdf conversion will be applied to the given svg

import lxml.etree
import sys
import os
import subprocess
import re


def main():
    import warnings
    # HIDE DEPRECATION WARINGS ONLY IN RELEASES. SHOW THEM IN DEV. TRUNKS
    warnings.filterwarnings('ignore', category=DeprecationWarning)

    if len(sys.argv) == 2:
        PATH = sys.argv[1]
        OUTDIR = os.getcwd()

    elif len(sys.argv) == 3:
        PATH = sys.argv[2]
        OUTDIR = sys.argv[1]
        
    else:
        print("Usage: %s [OUTPUT DIRECTORY] SVGFILE" % sys.argv[0])
        sys.exit(1)
    
    DIRNAME, FILENAME = os.path.split(PATH)
    BASENAME = os.path.splitext(FILENAME)[0]
    
    # Take the Wireframes.svg
    f = open(PATH)
    cnt = f.read().encode('utf-8')
    f.close()

    doc = lxml.etree.fromstring(cnt)

    # Get all layers
    layers = [x for x in doc.iterdescendants(tag='{http://www.w3.org/2000/svg}g') if x.attrib.get('{http://www.inkscape.org/namespaces/inkscape}groupmode', False) == 'layer']

    # Scan the 'content' layer
    content_layer = [x for x in layers if x.attrib.get('{http://www.inkscape.org/namespaces/inkscape}label', False).lower() == 'content']

    if not content_layer:
        svgslide="%s/%s.svg" % (DIRNAME,BASENAME)
        pdfslide="%s/%s.pdf" % (OUTDIR,BASENAME)

        output = str(subprocess.check_output("inkscape1.3 --version", shell=True, stderr=open(os.devnull, 'w')))

        if "Inkscape 0." in output:
            print("Inkscape Version = 0.*")
            version = 0
        else:
            print("Inkscape Version = 1.*")
            version = 1

        if version == 0:
            # for inscape 0.*
            os.system("inkscape -d 90 -C -A %s %s" % (pdfslide, svgslide))
        else:
            # for inkscape 1.x
            os.system("inkscape -d 90 -C --export-type=pdf --export-filename=%s %s" % (pdfslide, svgslide))

        # os.system("inkscape %s --export-pdf=%s" % (svgslide, pdfslide))
        sys.exit(0)

    content = content_layer[0]

    # Find the text stuff, everything starting with SLIDE:
    #   take all the layer names separated by ','..
    preslides = [x.text for x in content.findall('{http://www.w3.org/2000/svg}text/{http://www.w3.org/2000/svg}tspan') if x.text]


    if not bool(preslides):
        print("Make sure you have a text box (with no flowRect) in the " \
              "'content' layer, and rerun this program.")
        sys.exit(1)


    # Get the initial style attribute and keep it
    orig_style = {}
    for l in layers:
        label = l.attrib.get('{http://www.inkscape.org/namespaces/inkscape}label') 
        if 'style' not in l.attrib:
            l.set('style', '')
        # Save initial values
        orig_style[label] = l.attrib['style']


    prefix = ""
    prefixes = {}
    filenames = []
    slides = [] # Contains seq of [('layer', opacity), ('layer', opacity), ..]
    for sl in preslides:
        if sl:
            if sl.startswith('+'):
                sl = sl[1:]
                sl_layers = slides[-1].copy()
            else:
                sl_layers = {}
                
            sl = sl.split(':')
            if len(sl) == 1:
                sl = sl[0]
            elif len(sl) == 2:
                # First Part is prefix name
                prefix = sl[0]
                sl = sl[1]
            else:
                print("Too many prefix separators ':'")
                sys.exit(1)

            for layer in sl.split(','):
                elements = layer.strip().split('*')
                name = elements[0].strip()
                opacity = None
                if len(elements) == 2:
                    opacity = float(elements[1].strip())
                sl_layers[name] = {'opacity': opacity}
            slides.append(sl_layers)
            counter = prefixes.setdefault(prefix, 0)
            filename = "%s_%s_p%02d" % (BASENAME, prefix, counter)
            filenames.append(filename)
            prefixes[prefix] = counter + 1

    def set_style(el, style, value):
        """Set the display: style, add it if it isn't there, don't touch the
        rest
        """
        if re.search(r'%s: ?[a-zA-Z0-9.]*' % style, el.attrib['style']):
            el.attrib['style'] = re.sub(r'(.*%s: ?)([a-zA-Z0-9.]*)(.*)' % style,
                                        r'\1%s\3' % value, el.attrib['style'])
        else:
            el.attrib['style'] = '%s:%s;%s' % (style, value, el.attrib['style'])


    pdfslides = []
    for i, (slide_layers, filename) in enumerate(zip(slides, filenames)):
        for l in layers:
            label = l.attrib.get('{http://www.inkscape.org/namespaces/inkscape}label')
            # Set display mode to original
            l.set('style', orig_style[label])

            # Don't show it by default...
            set_style(l, 'display', 'none')

            if label in slide_layers:
                set_style(l, 'display', 'inline')
                opacity = slide_layers[label]['opacity']
                if opacity:
                    set_style(l, 'opacity', str(opacity))
            #print l.attrib['style']
        svgslide = os.path.abspath(os.path.join(OUTDIR,
                                                "%s.svg" % filename))
        pdfslide = os.path.abspath(os.path.join(OUTDIR,
                                                "%s.pdf" % filename))
        # Write the XML to file, "wireframes.p1.svg"
        f = open(svgslide, 'w')
        f.write(lxml.etree.tostring(doc,encoding='unicode'))
        f.close()

        # Run inkscape -A wireframes.p1.pdf wireframes.p1.svg
        # First, check version
        output = str(subprocess.check_output("inkscape --version", shell=True, stderr=open(os.devnull, 'w')))

        if output.startswith('Inkscape 0'):
            version = 0
        else:
            version = 1

        if version == 0:
            # for inscape 0.*
            os.system("inkscape -d 90 -C -A %s %s" % (pdfslide, svgslide))
        else:
            # for inkscape 1.x
            os.system("inkscape -d 90 -C --export-type=pdf --export-filename=%s %s" % (pdfslide, svgslide))

        os.unlink(svgslide)
        pdfslides.append(pdfslide)

        print("Generated page %d." % (i+1))

if __name__ == "__main__":
    main()

