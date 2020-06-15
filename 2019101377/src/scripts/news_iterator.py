# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import tensorflow as tf
import numpy as np




class NewsIterator(object):
    

    def __init__(self, hparams, npratio=0, col_spliter=" ", ID_spliter="%"):
        
        self.col_spliter = col_spliter
        self.ID_spliter = ID_spliter
        self.batch_size = hparams.batch_size
        self.doc_size = hparams.doc_size
        self.his_size = hparams.his_size
        self.npratio = npratio

    def parser_one_line(self, line):
        
        words = line.strip().split(self.ID_spliter)

        cols = words[0].strip().split(self.col_spliter)
        label = [float(i) for i in cols[: self.npratio + 1]]
        candidate_news_index = []
        click_news_index = []
        imp_index = []
        user_index = []
        c_input_mask=[]
        c_segement=[]
        h_input_mask=[]
        h_segement=[]
        h_len=[]
        c_len=[]
        all_his=[]
        all_can=[]

        for news in cols[self.npratio + 1 :]:
            tokens = news.split(":")
            if "Impression" in tokens[0]:
                imp_index.append(int(tokens[1]))
            elif "User" in tokens[0]:
                user_index.append(int(tokens[1]))
            elif "CandidateNews" in tokens[0]:
                # word index start by 0
                w_temp=[int(i) for i in tokens[1].split(",")]
                candidate_news_index.append(w_temp)
                count0=w_temp.count(0)
                c_input_mask.append([1]*(len(w_temp)-count0)+[0]*count0)
                c_segement.append([0]*len(w_temp))

            elif "ClickedNews" in tokens[0]:
                w_temp=[int(i) for i in tokens[1].split(",")]
                click_news_index.append(w_temp)
                count0=w_temp.count(0)
                h_input_mask.append([1]*(len(w_temp)-count0)+[0]*count0)
                h_segement.append([0]*len(w_temp))
            elif "Hislen" in tokens[0]:
                h_len=[[int(i)] for i in tokens[1].split(",")]
                if len(h_len)!=50:
                    print(len(h_len),h_len)
                assert len(h_len)==50
                
            elif "Canlen" in tokens[0]:
                c_len=[[int(i)] for i in tokens[1].split(",")]
                #assert len(c_len)==5

            elif "Allhis" in tokens[0]:
                all_his=[int(i) for i in tokens[1].split(",")]
                #.append(w_temp)

            elif "Allcan" in tokens[0]:
                all_can=[int(i) for i in tokens[1].split(",")]
                #.append(w_temp)
            else:
                raise ValueError("data format is wrong")

        return (label, imp_index, user_index, candidate_news_index, click_news_index, c_input_mask,c_segement,h_input_mask,h_segement,h_len,c_len,all_his,all_can)

    def load_data_from_file(self, infile):
        
        label_list = []
        imp_indexes = []
        user_indexes = []
        candidate_news_indexes = []
        click_news_indexes = []
        cnt = 0
        #input_ids=[]
        c_input_masks=[]
        c_segements=[]
        h_input_masks=[]
        h_segements=[]
        h_len=[]
        c_len=[]
        all_his=[]
        all_can=[]

        with tf.gfile.GFile(infile, "r") as rd:
            for line in rd:

                (
                    label,
                    imp_index,
                    user_index,
                    candidate_news_index,
                    click_news_index,
                    c_input_mask,
                    c_segement,
                    h_input_mask,
                    h_segement,
                    h_len_t,
                    c_len_t,
                    all_his_t,
                    all_can_t,
                ) = self.parser_one_line(line)

                candidate_news_indexes.append(candidate_news_index)
                click_news_indexes.append(click_news_index)
                imp_indexes.append(imp_index)
                user_indexes.append(user_index)
                label_list.append(label)
                c_input_masks.append(c_input_mask)
                c_segements.append(c_segement)
                h_input_masks.append(h_input_mask)
                h_segements.append(h_segement)
                h_len.append(h_len_t)
                c_len.append(c_len_t)
                all_his.append(all_his_t)
                all_can.append(all_can_t)
                cnt += 1
                if cnt >= self.batch_size:
                    #input_mask=
                    

                    yield self._convert_data(
                        label_list,
                        imp_indexes,
                        user_indexes,
                        candidate_news_indexes,
                        click_news_indexes,
                        c_input_masks,
                        c_segements,
                        h_input_masks,
                        h_segements,
                        h_len,
                        c_len,
                        all_his,
                        all_can,
                    )
                    candidate_news_indexes = []
                    click_news_indexes = []
                    label_list = []
                    imp_indexes = []
                    user_indexes = []
                    #input_ids=[]
                    c_input_masks=[]
                    c_segements=[]
                    h_input_masks=[]
                    h_segements=[]
                    h_len=[]
                    c_len=[]
                    all_his=[]
                    all_can=[]
                    cnt = 0

    def _convert_data(
        self,
        label_list,
        imp_indexes,
        user_indexes,
        candidate_news_indexes,
        click_news_indexes,
        c_input_masks,
        c_segements,
        h_input_masks,
        h_segements,
        h_length,
        c_length,
        all_his,
        all_can


    ):
        

        labels = np.asarray(label_list, dtype=np.float32)
        imp_indexes = np.asarray(imp_indexes, dtype=np.int32)
        user_indexes = np.asarray(user_indexes, dtype=np.int32)
        candidate_news_index_batch = np.asarray(candidate_news_indexes, dtype=np.int32)
        click_news_index_batch = np.asarray(click_news_indexes, dtype=np.int32)
        #input_ids=np.asarray(input_ids, dtype=np.int32)
        c_input_masks=np.asarray(c_input_masks, dtype=np.int32)
        c_segements=np.asarray(c_segements, dtype=np.int32)
        h_input_masks=np.asarray(h_input_masks, dtype=np.int32)
        h_segements=np.asarray(h_segements, dtype=np.int32)
        # c_length=np.asarray([[[5]]*5]*64,dtype=np.int32)
        # h_length=np.asarray([[[5]]*50]*64,dtype=np.int32)
        # all_his=np.asarray([[10]]*64,dtype=np.int32)
        # all_can=np.asarray([[3]]*64)
        c_length=np.asarray(c_length,dtype=np.int32)
        h_length=np.asarray(h_length,dtype=np.int32)
        all_his=np.asarray(all_his,dtype=np.int32)
        all_can=np.asarray(all_can,dtype=np.int32)
        return {
            "impression_index_batch": imp_indexes,
            "user_index_batch": user_indexes,
            "clicked_news_batch": click_news_index_batch,
            "candidate_news_batch": candidate_news_index_batch,
            #"input_ids":input_ids,
            "c_input_masks":c_input_masks,
            "c_segments":c_segements,
            "h_input_masks":h_input_masks,
            "h_segments":h_segements,
            "labels": labels,
            "c_length":c_length,
            "h_length":h_length,
            "all_his":all_his,
            "all_can":all_can,
        }
