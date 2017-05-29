# skip-gram
word embeddingの学習手法の一つであるskip-gramをフルスクラッチで実装しました。

## 使い方
  0. skip-gramディレクトリに移動してください。
  1. preparate_courpas.py内のcourpas_PATH = '/path/to/corpus/neko.txt'を、自分のコーパスのパスに書き換えてください。
  2. train_word_embedding.pyを起動してください。
  3. パラメータを入力してください。全てを入力すると学習が始まります。なんの工夫もなく実装しているのでめちゃめちゃ遅いです。頑張って待ちましょう。
  4. 学習が終わったら、w2v.pyを起動し、モデルにしたいエポック数を入力してください。
  5. 単語を入れると、その単語に意味の近い上位9件が表示されます。
  
