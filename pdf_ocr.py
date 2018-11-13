from PIL import Image
from wand.image import Image as wandImage
from wand.color import Color
from mrjob.job import MRJob
import cv2
import PyPDF2
import pytesseract
import argparse
import io
import numpy as np
import pandas as pd
import tempfile
import pickle
import boto3

parser = argparse.ArgumentParser(description='Use tesseract to process a pdf.')
parser.add_argument('filename', help='Name of pdf file to parse')
parser.add_argument('pagenum', type=int, help='page number to extract and parse')
args = parser.parse_args()


#args = {"filename": "1863Langley/1863Langley.pdf", "pagenum": 360, "resolution": 300, "preprocess": "thresh"}

def pdf_page_to_img(filename, pagenum, resolution=300, preprocess="thresh"):

    src_pdf = PyPDF2.PdfFileReader(filename)
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

def ocr(image, verbose=True, debug=False):
    # load the image as a PIL/Pillow image, apply OCR
    pagecontents = {}
    data = pytesseract.image_to_data(image, output_type="df.frame")
    boxes = pytesseract.image_to_boxes(image)
    try:
        df = pd.read_csv(io.StringIO(data), sep="\t")
        df = df.loc[(list(map(lambda x: str(x).strip()!='', df['text'])) & (df['conf']> -1) & (df['text']).notnull())]
        df = df.set_index(["block_num","par_num"])
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
    except:
        print("exception")

    if debug == True:
        tmp = tempfile.mkstemp(suffix='.png', prefix='tmp', dir=None, text=False)
        imgfilename = tmp[1]
        print(imgfilename)
        pagecontents['filename'] = imgfilename
        image.save(imgfilename)

    if verbose == True:
        df = pd.read_csv(io.StringIO(data), sep="\t")
        df = df.loc[(list(map(lambda x: str(x).strip()!='', df['text'])) & (df['conf']> -1) & (df['text']).notnull())]
        pagecontents['data'] = df
        #pagecontents['boxes'] = pd.read_csv(io.StringIO(boxes), sep="\s", header=None, names=["character", "x1","y1","x2","y2","c"])

    return pagecontents

contents = ocr(pdf_page_to_img(args.filename, args.pagenum))
pickle.dump( contents, open( "contents.p", "wb" ) )

# with open('contents.p', 'rb') as pickle_file:
#     c = pickle.load(pickle_file)
