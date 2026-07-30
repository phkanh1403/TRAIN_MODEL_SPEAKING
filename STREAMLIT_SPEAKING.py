import streamlit as st
import numpy as np
import pandas as pd

df = pd.read_csv(r'D:\TRAIN_MODEL_SPEAKING\SPEAKING_DATASET.csv')
st.title("NHẬP DỮ LIỆU SPEAKING IELTS")
tab1,tab2 = st.tabs(["THÊM","SỬA"])
with tab1:
    st.header("THÊM DỮ LIỆU")
    question = st.text_area("Nhập câu hỏi")
    answer = st.text_area("Nhập câu trả lời")
    context = st.text_input("Nhập chủ đề")
    col1,col2,col3 = st.columns(3)
    with col1:
        score = st.number_input("Nhập điểm")
    with col2:
        id_cluster = st.number_input("Nhập stt")
    with col3:
        part = st.number_input("Nhập phần")
    if st.button('Hoàn thành'):
        id = len(df) + 1
        id = str(int(id))+"."+str(int(id_cluster))+"."+str(int(part))
        new_row = {
            'ID':id,
            'ID_CLUSTER':id_cluster,
            'QUESTION':question,
            'ANSWER':answer,
            'CONTEXT':context,
            'CONTEXT_CLUSTER':part,
            'SCORE':score
        }
        df.loc[len(df)] = new_row
        df.to_csv(r'D:\TRAIN_MODEL_SPEAKING\SPEAKING_DATASET.csv',index=False,encoding="utf-8-sig")
        st.write('Đã nhập')
with tab2:
    st.header("SỬA DỮ LIỆU")
    col4,col5,col6 = st.columns(3)
    with col4:
        id_fix_input = st.number_input("Nhập dòng sửa")
    with col5:
        id_cluster_fix = st.number_input("Nhập stt sửa")
    with col6:
        part_fix = st.number_input("Nhập phần sửa")
    id_fix = str(int(id_fix_input))+"."+str(int(id_cluster_fix))+"."+str(int(part_fix))

    st.subheader("NHẬP DỮ LIỆU CẦN SỬA")
    part_fix_new = st.number_input("Nhập phần sửa lại")
    id_cluster_fix_new = st.number_input("Nhập stt sửa lại")
    question_fix = st.text_area("Nhập câu hỏi sửa")
    answer_fix = st.text_area("Nhập câu trả lời sửa")
    context_fix = st.text_input("Nhập chủ đề sửa")
    score_fix = st.number_input("Nhập điểm sửa")
    if st.button("Sửa"):
        id_fix_new = str(int(id_fix_input))+"."+str(int(id_cluster_fix_new))+"."+str(int(part_fix_new))
        mask = df["ID"] == id_fix
        df.loc[mask, "ID"] = id_fix_new
        df.loc[mask, "QUESTION"] = question_fix
        df.loc[mask, "ANSWER"] = answer_fix
        df.loc[mask, "CONTEXT"] = context_fix
        df.loc[mask, "SCORE"] = score_fix
        df.loc[mask, "CONTEXT_CLUSTER"] = part_fix_new
        df.loc[mask, "ID_CLUSTER"] = id_cluster_fix_new
        df.to_csv(
            r"D:\TRAIN_MODEL_SPEAKING\SPEAKING_DATASET.csv",index=False,encoding="utf-8-sig")
        st.write("Đã sửa")
        st.dataframe(df[mask])

