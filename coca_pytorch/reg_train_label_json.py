merged_diagnosis_dic = {
    # 恶性肿瘤 - 浸润性癌
    "Invasive carcinoma (NST)": 905,
    "Micro-invasive carcinoma": 6,
    "Invasive lobular carcinoma": 76,
    "Invasive carcinoma with mucinous features": 10,
    "Invasive micropapillary carcinoma": 8,
    "Mucinous carcinoma": 41,
    "Metaplastic carcinoma": 4,
    "Invasive cribriform carcinoma": 2,
    "Tubular carcinoma": 2,
    "Carcinoma with apocrine differentiation": 1,
    "Malignant lymphoma": 6,

    # 原位癌
    "Ductal carcinoma in situ (DCIS)": 282 + 2 + 150 + 16 + 18 + 1 + 1 + 2 + 2,
    # DCIS + intraductal papilloma(2) + papillary neoplasm(150) + intraductal papilloma(16) + intraductal papilloma with UDH(18) + intraductal papilloma with apocrine metaplasia(1) + papillary neoplasm with apocrine metaplasia(1) + solid papillary carcinoma in situ(2) + papillary neoplasm with ADH(2)
    "Lobular carcinoma in situ (LCIS)": 5,

    # 良性肿瘤
    "Fibroepithelial tumor (incl. fibroadenoma, phyllodes)": 91 + 9 + 4 + 19,
    # fibroepithelial tumor + fibroadenomatoid change + phyllodes tumor + fibroadenoma
    "Papillary neoplasm (benign)": 150,  # 已在DCIS里算过恶性部分

    # 良性病变
    "Columnar cell lesions": 36 + 1 + 1,  # columnar cell lesion + change + hyperplasia
    "Usual ductal hyperplasia": 23,
    "Atypical ductal hyperplasia": 32,
    "Flat epithelial atypia": 5,
    "Fibrocystic change": 35,
    "Duct ectasia": 14,
    "Granulomatous lobular mastitis": 3,
    "Sclerosing adenosis": 18,
    "Fat necrosis": 2,
    "Pseudoangiomatous stromal hyperplasia": 7,
    "Apocrine lesions": 3 + 3,  # apocrine metaplasia + apocrine adenosis
    "Foreign body reaction": 6,
    "Desmoid fibromatosis": 4,

    # 阴性
    "No evidence of tumor": 34,
}
