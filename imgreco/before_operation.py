import sys
from functools import lru_cache

import numpy as np
from PIL import Image

from richlog import get_logger
from . import imgops
from . import minireco
from . import resources
from . import util

LOGFILE = 'b4op.html'


@lru_cache(1)
def load_data():
    reco = minireco.MiniRecognizer(resources.load_pickle('minireco/NotoSansCJKsc-Medium.dat'))
    reco2 = minireco.MiniRecognizer(resources.load_pickle('minireco/Novecentosanswide_Medium.dat'))
    return (reco, reco2)


def recognize(img):
    logger = get_logger(LOGFILE)
    vw, vh = util.get_vwvh(img.size)

    apimg = img.crop((87.4 * vw, 3.3 * vh, 91.2 * vw, 7.2 * vh)).convert('L')
    reco_Noto, reco_Novecento = load_data()
    apimg = imgops.enhance_contrast(apimg, 80, 255)
    logger.logimage(apimg)
    aptext = reco_Noto.recognize(apimg)
    logger.logtext(aptext)
    # print("AP:", aptext)

    opidimg = img.crop((71.2*vw, 11.8 * vh, 76.7 * vw, 15 * vh)).convert('L')
    opidimg = imgops.enhance_contrast(opidimg, 80, 255)
    logger.logimage(opidimg)
    opidtext = reco_Novecento.recognize(opidimg)
    if opidtext.endswith('-'):
        opidtext = opidtext[:-1]
    opidtext = opidtext.upper()
    logger.logtext(opidtext)
    # print('operation:', opidtext)

    delegateimg = img.crop((82.7 * vw, 81.2 * vh, 83.5 * vw, 83.1 * vh)).convert('L')
    logger.logimage(delegateimg)
    score = np.count_nonzero(np.asarray(delegateimg) > 127) / (delegateimg.width * delegateimg.height)
    delegated = score > 0.5
    # print('delegated:', delegated)

    consumeimg = img.crop((91.6 * vw, 94.5 * vh, 93.1 * vw, 97 * vh)).convert('L')
    consumeimg = imgops.enhance_contrast(consumeimg, 80, 255)
    logger.logimage(consumeimg)
    consumetext = reco_Noto.recognize(consumeimg)
    consumetext = ''.join(c for c in consumetext if c in '0123456789')
    logger.logtext(consumetext)

    return {
        'AP': aptext,
        'operation': opidtext,
        'delegated': delegated,
        'consume': int(consumetext) if consumetext.isdigit() else None
    }
    # print('consumption:', consumetext)


def get_delegate_rect(viewport):
    vw, vh = util.get_vwvh(viewport)
    return (100 * vw - 32.083 * vh, 79.907 * vh, 100 * vw - 5.972 * vh, 84.444 * vh)


def get_start_operation_rect(viewport):
    vw, vh = util.get_vwvh(viewport)
    return (1930, 947, 2195, 1009)


def check_confirm_troop_rect(img):
    vw, vh = util.get_vwvh(img.size)
    icon1 = img.crop((50 * vw + 57.083 * vh, 64.722 * vh, 50 * vw + 71.389 * vh, 79.167 * vh)).convert('RGB')
    icon2 = resources.load_image_cached('before_operation/operation_start.png', 'RGB')
    icon1, icon2 = imgops.uniform_size(icon1, icon2)
    mse = imgops.compare_ccoeff(np.asarray(icon1), np.asarray(icon2))
    print(mse)
    return mse > 0.9


def get_confirm_troop_rect(viewport):
    vw, vh = util.get_vwvh(viewport)
    return (1788, 588, 1941, 918)


if __name__ == "__main__":
    print(recognize(Image.open(sys.argv[-1])))
