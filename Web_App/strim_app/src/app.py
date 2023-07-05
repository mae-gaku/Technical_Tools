import streamlit as st
from io import BytesIO
from zipfile import ZipFile
from PIL import Image
import glob


# ページタイトルの設定
st.set_page_config(page_title="Zipファイル解凍アプリ")

# ファイルのアップロードを受け付ける
uploaded_file = st.file_uploader("Choose a file", type="zip")

if uploaded_file is not None:
    # zipファイルをメモリ上で解凍する
    with ZipFile(BytesIO(uploaded_file.read())) as zip:
        zip.extractall()
    
    # 画像ファイルをリサイズする
    file = glob.glob("./src/dataset")
    img = Image.open('image.jpg')
    img_resized = img.resize((img.width//2, img.height//2))

    # リサイズされた画像を表示する
    st.image(img_resized)


else:
    st.info("Zipファイルをアップロードしてください。")



st.sidebar.text_input("文字入力欄")
st.sidebar.text_area("テキストエリア")

st.sidebar.checkbox("チェックボックス") #引数に入れることでboolを返す
st.sidebar.button("ボタン") #引数に入れるとboolで返す
st.sidebar.selectbox("メニューリスト", ("選択肢1", "選択肢2", "選択肢3")) #第一引数：リスト名、第二引数：選択肢
st.sidebar.multiselect("メニューリスト（複数選択可）", ("選択肢1", "選択肢2", "選択肢3")) #第一引数：リスト名、第二引数：選択肢、複数選択可
st.sidebar.radio("ラジオボタン", ("選択肢1", "選択肢2", "選択肢3")) #第一引数：リスト名（選択肢群の上に表示）、第二引数：選択肢

