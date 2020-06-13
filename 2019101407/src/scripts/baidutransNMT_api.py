import time
import pickle
import random
from http.client import HTTPConnection
import hashlib
import urllib
import random
import json
import re
import os


#用于将外文知识图谱的信息翻译成中文的
#这里使用的是百度的NMT模型的api
#翻译完的结果中,在本次实验仅使用名字部分来学习语义表示，其他部分暂不使用

#return result_list,flag.
def baiduapi(text,from_lang,to_lang):
    appid = 'xxxxxxx' #保密起见删掉了
    secretKey = 'xxxxxxxx'

    httpClient = None
    myurl = '/api/trans/vip/translate'
    q = text
    fromLang = from_lang
    toLang = to_lang
    salt = random.randint(32768, 65536)

    sign = appid + q + str(salt) + secretKey
    # m1 = md5.new()
    m1 = hashlib.md5()
    m1.update(sign.encode())
    sign = m1.hexdigest()
    myurl = myurl + '?appid=' + appid + '&q=' + urllib.parse.quote(q) + '&from=' + fromLang + '&to=' + toLang + '&salt=' + str(salt) + '&sign=' + sign

    try:
        httpClient = HTTPConnection('api.fanyi.baidu.com')
        httpClient.request('GET', myurl)
        # response是HTTPResponse对象
        response = httpClient.getresponse()
        get_vaule = response.read().decode('utf-8')
        get_vaule = json.loads(get_vaule)

        result = get_vaule['trans_result']
        res = []
        for one in result:
            res.append(one['dst'])
        flag = True

    except Exception as e:
        print(e)
        flag = False
        res = []

    finally:
        if httpClient:
            httpClient.close()
        return res,flag


def lower_rel_name(name):
    if r"/property/" in name:
        a, b = name.split(r"/property/")
        b = b.lower()
        string = a + r"/property/" + b
    else:
        a, b = name.rsplit(r'/',1)
        b = b.lower()
        string = a + r"/" + b
    return string



def read_rel_part(path):
    #rel part is with code part.
    ents_1 = []
    ents_2 = []
    ent2name_1 = dict()
    ent2name_2 = dict()
    rels_1 = []
    rels_2 = []
    rel2name_1 = dict()
    rel2name_2 = dict()
    triples_1 = []
    triples_2 = []
    ent_ills = []

    with open(path + "ent_ids_1","r",encoding="utf-8") as f:
        for line in f:
            id, name = line.rstrip("\n").split('\t')
            id = int(id)
            ent2name_1[id] = name
            ents_1.append(name)
    with open(path + "ent_ids_2","r",encoding="utf-8") as f:
        for line in f:
            id, name = line.rstrip("\n").split('\t')
            id = int(id)
            ent2name_2[id] = name
            ents_2.append(name)
    with open(path + "rel_ids_1","r",encoding="utf-8") as f:
        for line in f:
            id , name = line.rstrip("\n").split('\t')
            name = lower_rel_name(name)
            id = int(id)
            rel2name_1[id] = name
            rels_1.append(name)
    with open(path + "rel_ids_2","r",encoding="utf-8") as f:
        for line in f:
            id , name = line.rstrip("\n").split('\t')
            name = lower_rel_name(name)
            id = int(id)
            rel2name_2[id] = name.lower()
            rels_2.append(name)
    with open(path + "triples_1","r",encoding="utf-8") as f:
        for line in f:
            h,r,t = line.rstrip("\n").split('\t')
            h = ent2name_1[int(h)]
            r = rel2name_1[int(r)]
            t = ent2name_1[int(t)]
            triples_1.append((h,r,t))
    with open(path + "triples_2","r",encoding="utf-8") as f:
        for line in f:
            h,r,t = line.rstrip("\n").split('\t')
            h = ent2name_2[int(h)]
            r = rel2name_2[int(r)]
            t = ent2name_2[int(t)]
            triples_2.append((h,r,t))
    with open(path + "ent_ILLs","r",encoding="utf-8") as f:
        for line in f:
            ent1,ent2 = line.rstrip("\n").split('\t')
            ent_ills.append((ent1,ent2))

    return triples_1,triples_2,ents_1,ents_2,rels_1,rels_2,ent_ills


def reform_literal(liter_string):
    return liter_string.rstrip('@zhenfrdeja .\n').strip("<>").split(r"^^<")[0].strip("\"").replace('\n',' ')

def reform_att(att_string):
    if r"/property/" in att_string:
        return att_string.split(r"/property/")[-1].lower()
    else:
        return att_string.split(r"/")[-1].lower()
def refrom_name(string):
    return string.split(r"dbpedia.org/resource/")[-1].lower()

# def sip_del_char2(string):
#     return string.replace('.', '').replace('(', '').replace(')', '').replace(',', '').replace('_', '').replace('-', '').replace(' ', '')

def dict_add_r_t(dic,h,r,t):
    if h not in dic:
        dic[h] = dict()
    if r not in dic[h]:
        dic[h][r] = set()
    dic[h][r].add(t)

def read_att_part(path,lan1,lan2,new_path,top_num=50):
    print("TOP ATT NUM IS :",top_num)
    triples_1 = []
    triples_2 = []
    re_att_triples_1 = []
    re_att_triples_2 = []
    atts_1 = []
    atts_2 = []
    h_a_v_1 =dict()
    h_a_v_2 =dict()


    with open(path + lan1 + "_att_triples","r",encoding="utf-8") as f:
        for line in f:
            string = line.rstrip("\n .")
            h,a,v = string.split(' ',2)
            h = h.strip("<>")
            a = lower_rel_name(a.strip("<>"))
            v = reform_literal(v.strip("<>"))
            triples_1.append((h,a,v))
            dict_add_r_t(h_a_v_1, h, a, v)
    with open(path + lan2 + "_att_triples","r",encoding="utf-8") as f:
        for line in f:
            string = line.rstrip("\n .")
            h,a,v = string.split(' ',2)
            h = h.strip("<>")
            a = lower_rel_name(a.strip("<>"))
            v = reform_literal(v.strip("<>"))
            triples_2.append((h,a,v))
            dict_add_r_t(h_a_v_2, h, a, v)


    #fre cumu
    fre_att_dict1 = dict()
    fre_att_dict2 = dict()
    for h in h_a_v_1.keys():
        for a in h_a_v_1[h].keys():
            if a not in fre_att_dict1:
                fre_att_dict1[a] = 0
            fre_att_dict1[a] += 1
    for h in h_a_v_2.keys():
        for a in h_a_v_2[h].keys():
            if a not in fre_att_dict2:
                fre_att_dict2[a] = 0
            fre_att_dict2[a] += 1


    #get top num st att
    att1_list = [(att,fre) for att,fre in fre_att_dict1.items()]
    att2_list = [(att,fre) for att,fre in fre_att_dict2.items()]
    att1_list.sort(key=lambda x:x[1],reverse=True)
    att2_list.sort(key=lambda x:x[1],reverse=True)
    re_att_1 = [att for att,fre in att1_list][:top_num]
    re_att_2 = [att for att,fre in att2_list][:top_num]
    re_att_1 = set(re_att_1)
    re_att_2 = set(re_att_2)
    print("print(len(re_att_1))     print(len(re_att_2))")
    print(len(re_att_1))
    print(len(re_att_2))



    #filter triples
    for h,a,v in triples_1:
        if a in re_att_1 and v.replace(' ','')!="":
            re_att_triples_1.append((h,a,v))
            atts_1.append(a)
    for h,a,v in triples_2:
        if a in re_att_2 and v.replace(' ','')!="":
            re_att_triples_2.append((h,a,v))
            atts_2.append(a)

    #write down new atttriples:
    with open(new_path + "new_att_triples_1","w",encoding="utf-8") as f:
        for h,a,v in re_att_triples_1:
            string = h + '\t' + a + '\t' + v + '\n'
            f.write(string)
    with open(new_path + "new_att_triples_2","w",encoding="utf-8") as f:
        for h,a,v in re_att_triples_2:
            string = h + '\t' + a + '\t' + v + '\n'
            f.write(string)

    return atts_1,atts_2,re_att_triples_1,re_att_triples_2

#get it from multiKE code.
def sip_del_char(string):
    return string.replace('.', '').replace('(', '').replace(')', '').replace(',', '').replace('_', ' ')#.replace('-', ' ')

def get_to_translate_vaule_list(ents_1,ents_2,rels_1,rels_2,atts_1,atts_2,att_triples_1,att_triples_2):
    eng_mapping_ent = dict()
    eng_mapping_rel = dict()

    vaule_list = []
    rel_name_list = []
    entity_name_list = []

    for ent_list in [ents_1]:
        for ent in ent_list:
            name = refrom_name(ent)
            eng_mapping_ent[ent] = name
            entity_name_list.append(name)#
    for ent_list in [ents_2]:
        for ent in ent_list:
            name = sip_del_char(refrom_name(ent))
            eng_mapping_ent[ent] = name
            vaule_list.append(name)
    for rel_list in [rels_1,atts_1]:
        for rel in rel_list:
            name = sip_del_char(reform_att(rel))
            eng_mapping_rel[rel] = name
            rel_name_list.append(name)
    for rel_list in [rels_2,atts_2]:
        for rel in rel_list:
            name = sip_del_char(reform_att(rel))
            eng_mapping_rel[rel] = name
            vaule_list.append(name)
    for att_tri_list in [att_triples_1,att_triples_2]:
        for h,a,v in att_tri_list:
            vaule_list.append(v)

    return vaule_list,eng_mapping_ent,eng_mapping_rel,entity_name_list,rel_name_list

def isen_string(string):
    temp_string = string.strip()
    temp_string = temp_string.replace('.', '').replace('(', '').replace(')', '').replace(',', '').replace('_', '').replace('-', '').replace(' ', '')
    temp_string = temp_string.replace(r'"', '').replace(r'/', '').replace('\\', '').replace('#', '').replace(':','').replace('?','').replace('!','')
    if re.match(pattern = '[0-9a-zA-Z_.-]+$',string=temp_string):
        return True
    else:
        return False



def write_rel_triples(path,rel_triple_1,rel_triples_2):
    for filename,rel_tri in [("rel_triples_1",rel_triple_1),("rel_triples_2",rel_triples_2)]:
        with open(path + filename,"w",encoding="utf-8") as f:
            for h,r,t in rel_tri:
                string = h + "\t" + r + "\t" + t + '\n'
                f.write(string)

def write_train_val_test_ill_pairs(path,valid_path,ills):
    all_data = ills
    with open(path+"ent_links","w",encoding="utf-8") as f:
        for e1,e2 in all_data:
            string = e1 + '\t' + e2 +'\n'
            f.write(string)

    train_data = random.sample(all_data,4500)
    all_data = list(set(all_data)-set(train_data))
    valid_data = random.sample(all_data,1000)
    test_data = list(set(all_data)-set(valid_data))
    print("train/vaild/test length:",len(train_data),len(valid_data),len(test_data))
    print("overloap:",len(set(train_data)&set(test_data)),len(set(train_data)&set(valid_data)),
          len(set(test_data)&set(valid_data)))

    temp_list = []
    temp_list.append(("train_links",train_data))
    temp_list.append(("valid_links",valid_data))
    temp_list.append(("test_links",test_data))

    for filename,data in temp_list:
        with open(valid_path+filename,"w",encoding="utf-8") as f:
            for e1,e2 in data:
                string = e1 + '\t' + e2 +'\n'
                f.write(string)


def write_predicate_local_name(path,rel2name,name2en,o_rels_1, o_rels_2,o_atts_1, o_atts_2):
    #predicate_local_name_1 predicate_local_name_1
    all_rel_1 = []
    all_rel_2 = []
    for rel in o_rels_1:
        all_rel_1.append(rel)
    for rel in o_rels_2:
        all_rel_2.append(rel)
    for rel in o_atts_1:
        all_rel_1.append(rel)
    for rel in o_atts_2:
        all_rel_2.append(rel)
    all_rel_1 = list(set(all_rel_1))
    all_rel_2 = list(set(all_rel_2))

    with open(path + "predicate_local_name_1","w",encoding="utf-8") as f:
        for rel in all_rel_1:
            name = rel2name[rel]
            en = name2en[name].lower()
            string = rel + '\t' + en + '\n'
            f.write(string)
    with open(path + "predicate_local_name_2","w",encoding="utf-8") as f:
        for rel in all_rel_2:
            name = rel2name[rel]
            en = name2en[name].lower()
            string = rel + '\t' + en + '\n'
            f.write(string)


def write_entity_name(path,ents_1,ents_2,ent2name,name2en):
    temp_list = []
    temp_list.append(("entity_local_name_1",ents_1))
    temp_list.append(("entity_local_name_2",ents_2))

    index_num = 0
    for filename,ents in temp_list:
        with open(path + filename,"w",encoding="utf-8") as f:
            for e in ents:
                name = ent2name[e]
                if name in name2en:
                    en = name2en[name].lower()
                else:
                    en = refrom_name(e)
                    print("here name error in ",e,"name:",name,"fin name:",en)
                    index_num += 1
                string = e + '\t' + en +'\n'
                f.write(string)
    print("error write entity name num:",index_num)


def write_att_triples_data(path,att_triples_1,att_triples_2,vaule2en):
    temp_list = []
    temp_list.append(("attr_triples_1",att_triples_1))
    temp_list.append(("attr_triples_2",att_triples_2))

    for filename,att_tri in temp_list:
        with open(path + filename,"w",encoding="utf-8") as f:
            for h,a,v in att_tri:
                en = vaule2en[v].lower()
                string = h + '\t' + a + '\t' + en + '\n'
                f.write(string)




if __name__ == '__main__':

    ori_data_path = r"./dataset/ja_en/"
    new_data_path = r"./dataset/ja_en/"
    langague_align_dict_path = ori_data_path + "vaule_mapping.pickle" #
    valid_data_path = new_data_path
    lang = "ja"
    trans_lang = "ja"
    print(ori_data_path)
    print(new_data_path)


    if not os.path.exists(new_data_path):
        os.makedirs(new_data_path)
    if not os.path.exists(valid_data_path):
        os.makedirs(valid_data_path)

    vaule2en = dict()
    if os.path.exists(langague_align_dict_path):
        print("There is the pretraind language map.")
        vaule2en = pickle.load(open(langague_align_dict_path, "rb"))
        print("load the pretrained language map")
        print("map length = ", len(vaule2en.keys()))


    #0 read data
    o_rel_triples_1, o_rel_triples_2, o_ents_1, o_ents_2, o_rels_1, o_rels_2, ent_ills = read_rel_part(ori_data_path)
    o_atts_1, o_atts_2, o_att_triples_1, o_att_triples_2 = read_att_part(ori_data_path,lang,"en",new_data_path,top_num=40)

    o_atts_1 = list(set(o_atts_1))
    o_atts_2 = list(set(o_atts_2))

    o_rels_1 = list(set(o_rels_1))
    o_rels_2 = list(set(o_rels_2))
    print(len(o_atts_1))
    print(len(o_atts_2))


    index_num = 0
    for a,b in ent_ills:
        if refrom_name(a) == refrom_name(b):
            index_num+=1
    print(index_num/len(ent_ills))
    vaule_list, entity2name, rel2name,temp_entity_name_list,rel_name_list = \
        get_to_translate_vaule_list(o_ents_1, o_ents_2, o_rels_1, o_rels_2, o_atts_1, o_atts_2, o_att_triples_1, o_att_triples_2)

    print("len vaule list",len(vaule_list))
    vaule_list = list(set(vaule_list))
    print("len set vaule list", len(vaule_list))

    ####dele no need translate data:
    need_translate_list = []
    for vaule in vaule_list:
        if isen_string(vaule):
            if vaule not in vaule2en:
                vaule2en[vaule] = vaule
        else:
            if vaule not in vaule2en:
                need_translate_list.append(vaule)
    print(len(need_translate_list))
    for vaule in rel_name_list:
        if vaule not in vaule2en:
            need_translate_list.append(vaule)

    for vaule in temp_entity_name_list:
        vaule2en[vaule] = vaule
    print("len need translate data", len(need_translate_list))
    print("len need translate data", len(need_translate_list))


    #1 step----translate.
    batch_length = 40


    for i in range(0,len(need_translate_list),batch_length):
        print("Now:---",i,r"/",len(need_translate_list))
        temp_list = need_translate_list[i:min(i+batch_length ,len(need_translate_list))]

        #string is equal to lines
        string = ""
        for one in temp_list:
            string += "\n"+one
        string = string.strip()

        #string = temp_list[0]

        trans_res_list,flag = baiduapi(string, trans_lang ,"en")
        ori_str_list = string.split('\n')

        if len(ori_str_list)!=len(trans_res_list):
            print(len(ori_str_list))
            print(len(trans_res_list))
            print(ori_str_list)
            print(trans_res_list)
            print("error?in list length")
            flag = False
        else:
            try:
                for j in range(len(ori_str_list)):
                    vaule2en[ori_str_list[j]] = trans_res_list[j]
            except Exception as e:
                print(e)
                print("ERROR  except")
            if flag == False:
                print("ERROR!")
                break
        #must sleep!
        time.sleep(1)

        if i % 5000 == 0:
            with open(langague_align_dict_path, "wb") as f:
                pickle.dump(vaule2en, f)

    ####save the mapping
    with open(langague_align_dict_path, "wb") as f:
        pickle.dump(vaule2en, f)


    #2 step get_new_data!

    """
    -------------
    """
    #exit(0)



    #rel_triples_1 rel_triples_2
    write_rel_triples(new_data_path,o_rel_triples_1,o_rel_triples_2)
    #write ill
    write_train_val_test_ill_pairs(new_data_path,valid_data_path, ent_ills)
    #write predicate.
    write_predicate_local_name(new_data_path,rel2name,vaule2en,o_rels_1,o_rels_2,o_atts_1,o_atts_2)
    #write entity name
    write_entity_name(new_data_path,o_ents_1,o_ents_2,entity2name,vaule2en)
    #write att triples.
    write_att_triples_data(new_data_path,o_att_triples_1,o_att_triples_2,vaule2en)












