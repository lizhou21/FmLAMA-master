import os
import sys
import json
import time
import logging
import modules.base_connector as base
import glob


def load_jsonl(filename):
    data = []
    for file in glob.glob(str(filename)):
        with open(file, "r") as f:
            data += [json.loads(line) for line in f]
    return data


def create_logdir_with_timestamp(base_logdir, modelname):
    timestr = time.strftime("%Y%m%d_%H%M%S")

    # create new directory
    log_directory = "{}/{}_{}/".format(base_logdir, modelname, timestr)
    os.makedirs(log_directory)

    path = "{}/last".format(base_logdir)
    try:
        os.unlink(path)
    except Exception:
        pass
    os.symlink(log_directory, path)
    return log_directory


def init_logging(log_directory):
    logger = logging.getLogger("LAMA")
    logger.setLevel(logging.DEBUG)

    os.makedirs(log_directory, exist_ok=True)

    # logging format
    # "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # file handler
    fh = logging.FileHandler(str(log_directory) + "/info.log")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)

    # console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.WARNING)
    ch.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)

    logger.propagate = False

    return logger


def lowercase_samples(samples):
    new_samples = []
    for sample in samples:
        sample["obj_label"] = sample["obj_label"].lower()
        sample["sub_label"] = sample["sub_label"].lower()
        lower_masked_sentences = []
        for sentence in sample["masked_sentence"]:
            sentence = sentence.lower()
            sentence = sentence.replace(base.MASK.lower(), base.MASK)
            lower_masked_sentences.append(sentence)
        sample["masked_sentence"] = lower_masked_sentences

        new_samples.append(sample)
    return new_samples


def batchify(data, batch_size):
    # sort to group together sentences with similar length
    data = sorted(data, key=lambda k: len(" ".join(k["masked_sentence"]).split()))

    # Split data into batches
    list_samples_batches = [
        data[i : i + batch_size] for i in range(0, len(data), batch_size)
    ]

    return list_samples_batches


def fill_template_with_values(language, template, subject_label, object_label, origin, relation_name):
    """Fill template with a subject/object from a triple"""

    if relation_name.startswith("country"):
        if language=='he':
            template = template.replace("[3]", origin)
            template = template.replace("[1]", subject_label)
            template = template.replace("[2]", object_label)
        else:
            template = template.replace("[C]", origin)
            template = template.replace("[X]", subject_label)
            template = template.replace("[Y]", object_label)
    else:
        if language=='he':
            # template = template.replace("[3]", origin)
            template = template.replace("[1]", subject_label)
            template = template.replace("[2]", object_label)
        else:
            # template = template.replace("[C]", origin)
            template = template.replace("[X]", subject_label)
            template = template.replace("[Y]", object_label)

    return template



# English2Chinese = {
#     "People's Republic of China": "中国",
#     "Japan": "日本",
#     "United States of America": "美国",
#     "France": "法国",
#     "Italy": "意大利",
#     "United Kingdom": "英国",
#     "Korea": "韩国",
#     "India": "印度",
#     "Indonesia": "印度尼西亚",
#     "Turkey": "土耳其",
#     "Mexico": "墨西哥",
#     "Spain": "西班牙",
#     "Germany": "德国",
#     "Poland": "波兰",
#     "Russia": "俄罗斯",
#     "Sweden": "瑞典",
#     "Austria": "奥地利",
#     "Hungary": "匈牙利",
#     "Belgium": "比利时",
#     "Canada": "加拿大",
#     "Australia": "澳大利亚",
#     "Saudi Arabia": "沙特阿拉伯",
#     "South Africa": "南非",
#     "Argentina": "阿根廷",
#     "Peru": "秘鲁",
#     "Brazil": "巴西",
#     "Ethiopia": "埃塞俄比亚",
#     "Nigeria": "尼日利亚",
#     "Lithuania": "立陶宛",
#     "Chile": "智利",
#     "Jamaica": "牙买加",
#     "Colombia": "哥伦比亚",
#     "New Zealand": "新西兰",
# }

# English2English = {
#     "People's Republic of China": "China",
#     "Japan": "Japan",
#     "United States of America": " the United States",
#     "France": "France",
#     "Italy": "Italy",
#     "United Kingdom": "the United Kingdom",
#     "Korea": "South Korea",
#     "India": "India",
#     "Indonesia": "Indonesia",
#     "Turkey": "Turkey",
#     "Mexico": "Mexico",
#     "Spain": "Spain",
#     "Germany": "Germany",
#     "Poland": "Poland",
#     "Russia": "Russia",
#     "Sweden": "Sweden",
#     "Austria": "Austria",
#     "Hungary": "Hungary",
#     "Belgium": "Belgium",
#     "Canada": "Canada",
#     "Australia": "Australia",
#     "Saudi Arabia": "Saudi Arabia",
#     "South Africa": "South Africa",
#     "Argentina": "Argentina",
#     "Peru": "Peru",
#     "Brazil": "Brazil",
#     "Ethiopia": "Ethiopia",
#     "Nigeria": "Nigeria",
#     "Lithuania": "Lithuania",
#     "Chile": "Chile",
#     "Jamaica": "Jamaica",
#     "Colombia": "Colombia",
#     "New Zealand": "New Zealand",
# }



# English2Arabic = {
#    "People's Republic of China": "الصين",
#    "Japan": "اليابان",
#    "United States of America": "الولايات المتحدة",
#    "France": "فرنسا",
#    "Italy": "إيطاليا",
#    "United Kingdom": "المملكة المتحدة",
#    "Korea": "كوريا",
#    "India": "الهند",
#    "Indonesia": "إندونيسيا",
#    "Turkey": "تركيا",
#    "Mexico": "المكسيك",
#    "Spain": "إسبانيا",
#    "Germany": "ألمانيا",
#    "Poland": "بولندا",
#    "Russia": "روسيا",
#    "Sweden": "السويد",
#    "Austria": "النمسا",
#    "Hungary": "هنغاريا",
#    "Belgium": "بلجيكا",
#    "Canada": "كندا",
#    "Australia": "أستراليا",
#    "Saudi Arabia": "المملكة العربية السعودية",
#    "South Africa": "جنوب أفريقيا",
#    "Argentina": "الأرجنتين",
#    "Peru": "بيرو",
#    "Brazil": "البرازيل",
#    "Ethiopia": "أثيوبيا",
#    "Nigeria": "نيجيريا",
#    "Lithuania": "ليتوانيا",
#    "Chile": "شيلي",
#    "Jamaica": "جامايكا",
#    "Colombia": "كولومبيا",
#    "New Zealand": "نيوزيلندا",
# }

# English2Hebrew={
#     "People's Republic of China": "סין",
#     "Japan": "יפן",
#     "United States of America": "ארצות הברית",
#     "France": "צרפת",
#     "Italy": "איטליה",
#     "United Kingdom": "בריטניה",
#     "Korea": "קוריאה",
#     "India": "הודו",
#     "Indonesia": "אינדונזיה",
#     "Turkey": "טורקיה",
#     "Mexico": "מקסיקו",
#     "Spain": "ספרד",
#     "Germany": "גרמניה",
#     "Poland": "פולין",
#     "Russia": "רוסיה",
#     "Sweden": "שוודיה",
#     "Austria": "אוסטריה",
#     "Hungary": "הונגריה",
#     "Belgium": "בלגיה",
#     "Canada": "קנדה",
#     "Australia": "אוסטרליה",
#     "Saudi Arabia": "ערב הסעודית",
#     "South Africa": "דרום אפריקה",
#     "Argentina": "ארגנטינה",
#     "Peru": "פרו",
#     "Brazil": "ברזיל",
#     "Ethiopia": "אתיופיה",
#     "Nigeria": "ניגריה",
#     "Lithuania": "ליטואניה",
#     "Chile": "צ׳ילה",
#     "Jamaica": "ג׳מייקה",
#     "Colombia": "קולומביה",
#     "New Zealand": "ניו זילנד",
# }

# English2Korean={
#     "People's Republic of China": "중국",
#     "Japan": "일본",
#     "United States of America": "미국",
#     "France": "프랑스",
#     "Italy": "이탈리아",
#     "United Kingdom": "영국",
#     "Korea": "한국",
#     "India": "인도",
#     "Indonesia": "인도네시아",
#     "Turkey": "칠면조",
#     "Mexico": "멕시코",
#     "Spain": "스페인",
#     "Germany": "독일",
#     "Poland": "폴란드",
#     "Russia": "러시아",
#     "Sweden": "스웨덴",
#     "Austria": "오스트리아",
#     "Hungary": "헝가리",
#     "Belgium": "벨기에",
#     "Canada": "캐나다",
#     "Australia": "호주",
#     "Saudi Arabia": "사우디아라비아",
#     "South Africa": "남아프리카공화국",
#     "Argentina": "아르헨티나",
#     "Peru": "페루",
#     "Brazil": "브라질",
#     "Ethiopia": "에티오피아",
#     "Nigeria": "나이지리아",
#     "Lithuania": "리투아니아",
#     "Chile": "칠레",
#     "Jamaica": "자메이카",
#     "Colombia": "콜롬비아",
#     "New Zealand": "뉴질랜드"
# }

# English2Russian={
#    "People's Republic of China": "Китай",
#    "Japan": "Япония",
#    "United States of America": "Соединенные Штаты Америки",
#    "France": "Франция",
#    "Italy": "Италия",
#    "United Kingdom": "Великобритания",
#    "Korea": "Корея",
#    "India": "Индия",
#    "Indonesia": "Индонезия",
#    "Turkey": "Турция",
#    "Mexico": "Мексика",
#    "Spain": "Испания",
#    "Germany": "Германия",
#    "Poland": "Польша",
#    "Russia": "Россия",
#    "Sweden": "Швеция",
#    "Austria": "Австрия",
#    "Hungary": "Венгрия",
#    "Belgium": "Бельгия",
#    "Canada": "Канада",
#    "Australia": "Австралия",
#    "Saudi Arabia": "Судийская Аравия",
#    "South Africa": "Южная Африка",
#    "Argentina": "Аргентина",
#    "Peru": "Перу",
#    "Brazil": "Бразилия",
#    "Ethiopia": "Эфиопия",
#    "Nigeria": "Нигерия",
#    "Lithuania": "Литва",
#    "Chile": "Чили",
#    "Jamaica": "Ямайка",
#    "Colombia": "Колумбия",
#    "New Zealand": "Новая Зеландия",
# }