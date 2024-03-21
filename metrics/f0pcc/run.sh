mkdir TEST_
cd TEST_

ln -s ../../../test/TEST_SRC_VCTK src_vctk
ln -s ../../../test/TEST_SRC_LIBRI src_libri
ln -s ../../../test/TEST_SRC_ESD_EN src_esd_en
ln -s ../../../test/TEST_SRC_ESD_ZH src_esd_zh

ln -s ../../../outputs/s2s s2s
ln -s ../../../outputs/u2s u2s
ln -s ../../../outputs/esd_en esd_en
ln -s ../../../outputs/esd_zh esd_zh

cd ..


python pcc.py --wavdir TEST_/s2s --title s2s --srcdir TEST_/src_vctk
python pcc.py --wavdir TEST_/u2s --title u2s --srcdir TEST_/src_libri
python pcc.py --wavdir TEST_/esd_en --title esd_en --srcdir TEST_/src_esd_en
python pcc.py --wavdir TEST_/esd_zh --title esd_zh --srcdir TEST_/src_esd_zh
