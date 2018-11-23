# script to process vi tesseract ocr a single pdf file, writing the results into
# a pickle file. Writes to a temp file for each core then combines these into
# a single file, name given on the command line.

# ex:
# python pdf_ocr.py 1863Langley/1863Langley.pdf 1863Langley.p

from PIL import Image
from wand.image import Image as wandImage
from wand.color import Color
from mrjob.job import MRJob
import cv2
import csv
import PyPDF2
import pytesseract
import argparse
import io
import numpy as np
import pandas as pd
import tempfile
import pickle
import boto3
import multiprocessing

def pdf_page_to_img(src_pdf, pagenum, resolution=300, preprocess="thresh"):

    dst_pdf = PyPDF2.PdfFileWriter()
    dst_pdf.addPage(src_pdf.getPage(pagenum))
    pdf_bytes = io.BytesIO()
    dst_pdf.write(pdf_bytes)
    pdf_bytes.seek(0)
    img_buffer=None

    with wandImage(file = pdf_bytes, resolution = resolution) as img:
        img.format = "png"
        img.alpha_channel = False
        img_buffer=np.asarray(bytearray(img.make_blob()), dtype=np.uint8)

    if img_buffer is not None:
        retval = cv2.imdecode(img_buffer, cv2.IMREAD_UNCHANGED)
        image = cv2.cvtColor(retval, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    if preprocess == "thresh":
        gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    elif preprocess == "blur":
        gray = cv2.medianBlur(gray, 3)

    return Image.fromarray(gray)

def ocr(image, pagenum):
    # load the image as a PIL/Pillow image, apply OCR
    pagecontents = {}
    data = pytesseract.image_to_data(image, output_type="df.frame")
    boxes = pytesseract.image_to_boxes(image)
    try:
        df = pd.read_csv(io.StringIO(data), sep="\t", quoting=csv.QUOTE_NONE, encoding='utf-8', engine='python')
        dfm = df.loc[(list(map(lambda x: str(x).strip()!='', df['text'])) & (df['conf']> -1) & (df['text']).notnull())]
        df = dfm.set_index(["block_num","par_num"])
        blockstats = {'block':[], 'n_par':[], 'n_line':[], 'top_at':[], 'bottom_at':[], 'left_at':[], 'right_at':[]}
        paragraphs = {}
        for block in df.index.levels[0]:
            blockstats['block'].append(block)
            blockstats['n_par'].append(len(np.unique(df.loc[block,].index.values)))
            blockstats['n_line'].append(df.loc[(block,)]['top'].count())
            blockstats['top_at'].append(min(df.loc[(block,)]['top']))
            blockstats['bottom_at'].append(max(df.loc[(block,)]['top']+df.loc[(block,)]['height']))
            blockstats['left_at'].append(min(df.loc[(block,)]['left']))
            blockstats['right_at'].append(max(df.loc[(block,)]['left']+df.loc[(block,)]['width']))
            paragraphs[block] = []
            for i in df.loc[block].index.unique():
                paragraphs[block].append(' '.join(list(map(lambda x: x.strip(), df.loc[block].loc[i,'text']))))
        blockstats = pd.DataFrame(data=blockstats).set_index('block')
        pagecontents['blockstats'] = blockstats
        pagecontents['paragraphs'] = paragraphs
        pagecontents['data'] = dfm
        pagecontents['boxes'] = pd.read_csv(io.StringIO(boxes), quoting=csv.QUOTE_NONE, encoding='utf-8', engine='python',
                                                sep="\s", header=None, names=["character", "x1","y1","x2","y2","c"])
    except:
        print("exception: "+str(pagenum))
       
    pagecontents['pagenum'] = pagenum

    return pagecontents

def ocr_page(procnum,num_pages):
    proccontents = {}
    pagerange = range(procnum, num_pages, num_cpus)
    for pagenum in pagerange:
        page_img = pdf_page_to_img(src_pdf, pagenum)
        proccontents[pagenum] = ocr(page_img, pagenum)
        print("page [{0}]".format(pagenum))
        #q.put(contents)
    pickle.dump( proccontents, open( "contents."+str(procnum)+".p", "wb" ) )
    return proccontents

if __name__ == "__main__":

    # python pdf_ocr.py 1863Langley/1863Langley.pdf

    parser = argparse.ArgumentParser(description='Use tesseract to process a pdf.')
    parser.add_argument('filename', help='Name of pdf file to parse')
    parser.add_argument('outfilename', help='Name of output pickle file')
    args = parser.parse_args()
    
    src_pdf = PyPDF2.PdfFileReader(args.filename)
    num_pages = src_pdf.getNumPages()
    num_cpus = multiprocessing.cpu_count()

    contents = {}
    proccontents = {}
    pages = []

    processes = []

    for procnum in range(num_cpus):
        t = multiprocessing.Process(target=ocr_page, args=(procnum,num_pages,))
        processes.append(t)
        print("starting {0}".format(procnum))
        t.start()

    for p in processes:
        print("joining...")
        p.join()

    contents = {}
    for i in range(4):
        cont = pickle.load(open("contents."+str(i)+".p", "rb"))
        for kv in cont.keys():
            contents[kv] = cont[kv]

    pickle.dump( contents, open( args.outfilename + ".p", "wb" ) )
#    contents = pickle.load( open( "contents.p", "rb" ) )