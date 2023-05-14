import numpy as np
import re


class SWEEncoder_ja:
    def __init__(self, bpe, emoji) -> None:
        self.bpe = [[b] if (b == ',' or ',' not in b)
                    else b.split(',') for b in bpe]
        self.swe = {}
        for idx, b in enumerate(self.bpe):
            for wd in b:
                self.swe[wd] = idx
        self.emoji = emoji
        self.maxlen = np.max([len(w) for w in self.swe.keys()])
        self.content_repatter1 = re.compile(
            r"(https?|ftp)(:\/\/[-_\.!~*\'()a-zA-Z0-9;\/?:\@&=\+$,%#]+)")
        self.content_repatter2 = re.compile(
            r"[A-Za-z0-9\._+]*@[\-_0-9A-Za-z]+(\.[A-Za-z]+)*")
        self.content_repatter3 = re.compile(
            r'[\(]{0,1}[0-9]{2,4}[\)\-\(]{0,1}[0-9]{2,4}[\)\-]{0,1}[0-9]{3,4}')
        self.content_repatter4 = re.compile(
            r"([12]\d{3}[/\-年])*(0?[1-9]|1[0-2])[/\-月]((0?[1-9]|[12][0-9]|3[01])日?)*(\d{1,2}|:|\d{1,2}時|\d{1,2}分|\(日\)|\(月\)|\(火\)|\(水\)|\(木\)|\(金\)|\(土\)|㈰|㈪|㈫|㈬|㈭|㈮|㈯)*")
        self.content_repatter5 = re.compile(
            r"(明治|大正|昭和|平成|令和|㍾|㍽|㍼|㍻|\u32ff)\d{1,2}年(0?[1-9]|1[0-2])月(0?[1-9]|[12][0-9]|3[01])日(\d{1,2}|:|\d{1,2}時|\d{1,2}分|\(日\)|\(月\)|\(火\)|\(水\)|\(木\)|\(金\)|\(土\)|㈰|㈪|㈫|㈬|㈭|㈮|㈯)*")
        self.content_repatter6 = re.compile(
            r'((0|[1-9]\d*|[1-9]\d{0,2}(,\d{3})+)*億)*((0|[1-9]\d*|[1-9]\d{0,2}(,\d{3})+)*万)*((0|[1-9]\d*|[1-9]\d{0,2}(,\d{3})+)*千)*(0|[1-9]\d*|[1-9]\d{0,2}(,\d{3})+)*(千円|万円|千万円|円|千ドル|万ドル|千万ドル|ドル|千ユーロ|万ユーロ|千万ユーロ|ユーロ)+(\(税込\)|\(税抜\)|\+tax)*')
        keisen = "─━│┃┄┅┆┇┈┉┊┋┌┍┎┏┐┑┒┓└┕┖┗┘┙┚┛├┝┞┟┠┡┢┣┤┥┦┧┨┩┪┫┬┭┮┯┰┱┲┳┴┵┶┷┸┹┺┻┼┽┾┿╀╁╂╃╄╅╆╇╈╉╊╋╌╍╎╏═║╒╓╔╕╖╗╘╙╚╛╜╝╞╟╠╡╢╣╤╥╦╧╨╩╪╫╬╭╮╯╰╱╲╳╴╵╶╷╸╹╺╻╼╽╾╿"
        blocks = "▀▁▂▃▄▅▆▇█▉▊▋▌▍▎▏▐░▒▓▔▕▖▗▘▙▚▛▜▝▞▟"
        self.content_trans1 = str.maketrans(
            {k: '<BLOCK>' for k in keisen + blocks})

    def __len__(self):
        return len(self.bpe)  # トークン数を返す

    def clean_text(self, content):
        content = self.content_repatter1.sub("<URL>", content)
        content = self.content_repatter2.sub("<EMAIL>", content)
        content = self.content_repatter3.sub("<TEL>", content)
        content = self.content_repatter4.sub("<DATE>", content)
        content = self.content_repatter5.sub("<DATE>", content)
        content = self.content_repatter6.sub("<PRICE>", content)
        content = content.translate(self.content_trans1)
        while '<BLOCK><BLOCK>' in content:
            content = content.replace('<BLOCK><BLOCK>', '<BLOCK>')
        return content

    def encode(self, text, clean=False):
        """
        Inputs:
          text[str]: input string
          clean[bool]: cleaning string
          bpe_dropout_rate[float]:
            Flags to use when learning model. Whether to use BPE Dropout.
            The research to divides same word with different BPEs make efficiently learn low-frequency words.
            It is concluded that setting a value of 0.1 during learning is good.
            https://arxiv.org/abs/1910.13267
            https://ai-scholar.tech/articles/natural-language-processing/bpe-dropout
        Return:
          tokens[List[int]]
        """
        # テキストの前処理を行う　
        text = text.replace(' ', '<SP>')
        text = text.replace('　', '<SP>')
        text = text.replace('\r\n', '<BR>')
        text = text.replace('\n', '<BR>')
        text = text.replace('\r', '<BR>')
        text = text.replace('\t', '<TAB>')
        text = text.replace('—', 'ー')
        text = text.replace('−', 'ー')
        for k, v in self.emoji['emoji'].items():
            if k in text:
                text = text.replace(k, v)
        if clean:
            text = self.clean_text(text)

        def checkkigou(x):
            # 特殊トークン「記号」を判定する関数
            e = x.encode()
            if len(x) == 1 and len(e) == 2:  # 1文字で2バイトなら
                c = (int(e[0]) << 8) + int(e[1])  # 文字コードを作る
                if (c >= 0xc2a1 and c <= 0xc2bf) or (c >= 0xc780 and c <= 0xc783) or \
                        (c >= 0xcab9 and c <= 0xcbbf) or (c >= 0xcc80 and c <= 0xcda2):
                    return True  # 記号
            return False

        def checku2e(x):
            # 特殊トークン「3バイト記号」を判定する関数
            e = x.encode()
            if len(x) == 1 and len(e) == 3:  # 1文字で3バイトなら
                c = (int(e[0]) << 16) + (int(e[1]) << 8) + \
                    int(e[2])  # 文字コードを作る
                if c >= 0xe28080 and c <= 0xe2b07f:
                    return True  # 記号
            return False

        pos = 0
        result = []
        while pos < len(text):  # 現在の位置が終わりに達するまで
            # 文章の中の見る位置
            end = min(len(text), pos + self.maxlen +
                      1) if text[pos] == '<' else pos + 3

            kouho = []  # 一致した候補トークン
            # 一致するトークンを見つける
            for e in range(end, pos, -1):
                wd = text[pos:e]
                if wd in self.swe:
                    if wd[0] == '<' and len(wd) > 2:  # 特殊トークンの場合
                        kouho = [(self.swe[wd], e)]
                        break
                    else:
                        kouho.append((self.swe[wd], e))  # 通常トークンの場合
            if len(kouho) > 0:
                # トークンIDを取得
                wp, e = sorted(kouho, key=lambda x: x[0])[0]  # リストの一番上側
                result.append(wp)
                pos = e
            else:  # トークンのリストになかった
                end = pos + 1
                wd = text[pos:end]
                if checkkigou(wd):  # 特殊トークン「記号」
                    result.append(self.swe['<KIGOU>'])
                elif checku2e(wd):  # 特殊トークン「3バイト記号」
                    result.append(self.swe['<U2000U2BFF>'])
                else:
                    # 今見ている部分をバイト列にエンコード
                    for i in wd.encode('utf-8'):
                        # <|byte0|>〜<|byte255|>の列にエンコード
                        result.append(self.swe['<|byte%d|>' % i])
                pos = end  # 現在の位置を更新
        return result

    def decode(self, tokens, breakline='\n'):
        words = []
        byte_tokens = []
        for i in tokens:
            word = self.bpe[i][0]  # トークンのリストから文字列表現を得る
            if word[:6] == '<|byte' and word[-2:] == '|>':  # バイト列でエンコードされた
                byte_tokens.append(int(word[6:-2]))
            else:
                if len(byte_tokens) > 0:  # バイト列を元に戻す
                    words.append(bytearray(byte_tokens).decode(
                        'utf-8', errors='replace'))
                    byte_tokens = []
                # トークンから文字列を元に戻す
                if word[:7] == '<|emoji' and word[-2:] == '|>':
                    words.append(self.emoji['emoji_inv'][word])
                elif word == '<SP>':
                    words.append(' ')
                elif word == '<BR>':
                    words.append(breakline)
                elif word == '<TAB>':
                    words.append('\t')
                elif word == '<BLOCK>':
                    words.append('▀')
                elif word == '<KIGOU>':
                    words.append('ǀ')
                elif word == '<U2000U2BFF>':
                    words.append('‖')
                else:
                    words.append(word)
        if len(byte_tokens) > 0:  # バイト列を元に戻す
            words.append(bytearray(byte_tokens).decode(
                'utf-8', errors='replace'))
        text = ''.join(words)
        return text


if __name__ == '__main__':
    import argparse
    import shutil
    import os
    import json
    from tqdm import tqdm
    import pickle
    import uuid
    from multiprocessing import Pool

    # プログラムとして実行したときのオプション
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_dir", help="source dir", required=True)
    parser.add_argument("--dst_file", help="destination file", required=True)
    parser.add_argument("--tmp_dir", help="tempolary file", default="tmpfiles")
    parser.add_argument(
        "--vocabulary", help="vocabulary file", default="ja-swe32k.txt")
    parser.add_argument("--num_process", help="process num",
                        type=int, default=8)
    parser.add_argument(
        "--combine", help="Concatenate files with <|endoftext|> separator into chunks of this minimum size", type=int, default=50000)
    parser.add_argument('--clean_text', action='store_true')
    parser.add_argument(
        "--tmpsize", help="num chunks in tempolary file", type=int, default=5000)
    args = parser.parse_args()

    # 一時ディレクトリを作る
    if os.path.isdir(args.tmp_dir):
        shutil.rmtree(args.tmp_dir)
    os.mkdir(args.tmp_dir)

    # エンコーダーを作成する
    with open(args.vocabulary, encoding='utf-8') as f:
        bpe = f.read().split('\n')
    with open('emoji.json', encoding='utf-8') as f:
        emoji = json.loads(f.read())
    enc = SWEEncoder_ja(bpe, emoji)

    # 別プロセスで実行される関数
    array_file = []

    def _proc(i):
        token_chunks = []
        raw_text = ''
        # 対応するファイルのみ処理する
        for j, (curDir, dirs, files) in enumerate(array_file):
            if not (j % args.num_process == i):
                continue
            print('append #', curDir)
            for file in tqdm(files):
                if file.endswith(".txt"):
                    # ファイルをエンコードする
                    input = os.path.join(curDir, file)
                    with open(input, 'r', encoding='utf-8') as fp:
                        raw_text += fp.read()
                    raw_text += '<|endoftext|>'
                    # エンコード済みトークン列をつなげていく
                    if len(raw_text) >= args.combine:
                        tokens = np.stack(enc.encode(
                            raw_text, clean=args.clean_text))
                        token_chunks.append(tokens)
                        raw_text = ''
            if raw_text and len(raw_text) > 0:
                tokens = np.stack(enc.encode(raw_text))
                token_chunks.append(tokens)
            # メモリ中のエンコード済みデータが大きくなったら書き出す
            if len(token_chunks) > args.tmpsize:
                with open(os.path.join(args.tmp_dir, '%s.pkl' % str(uuid.uuid4())), 'wb') as f:
                    pickle.dump(token_chunks, f)
                    token_chunks = []
        # エンコード済みデータを書き出す
        with open(os.path.join(args.tmp_dir, '%s.pkl' % str(uuid.uuid4())), 'wb') as f:
            pickle.dump(token_chunks, f)

    # 処理するファイルが入ったディレクトリのリストを作る
    for curDir, dirs, files in os.walk(args.src_dir):
        array_file.append((curDir, dirs, files))

    # マルチプロセスですべて処理する
    with Pool(args.num_process) as p:
        p.map(_proc, list(range(args.num_process)))

    # 結果のファイルを読み込んでつなげる
    token_chunks = []
    for s in os.listdir(args.tmp_dir):
        with open(os.path.join(args.tmp_dir, s), 'rb') as f:
            token_chunks.extend(pickle.load(f))

    # 最終結果を保存する
    np.savez_compressed(args.dst_file, *token_chunks)
    shutil.rmtree(args.tmp_dir)
