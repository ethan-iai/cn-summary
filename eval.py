import jieba
from rouge import Rouge
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

class get_evaluation_index:

    def __init__(self):
        self.rouge = Rouge()
        self.smooth = SmoothingFunction().method1
    
    def get_rouge_1(self, source, target, unit='char'):
        if unit == 'word':
            source = jieba.cut(source)
            target = jieba.cut(target)
        source, target = ' '.join(source), ' '.join(target)
        score = self.rouge.get_scores(hyps=target, refs=source)
        rouge_1 = score[0]['rouge-1']['f']
        return rouge_1

    def get_rouge_2(self, source, target, unit='char'):
        if unit == 'word':
            source = jieba.cut(source)
            target = jieba.cut(target)
        source, target = ' '.join(source), ' '.join(target)
        score = self.rouge.get_scores(hyps=target, refs=source)
        rouge_2 = score[0]['rouge-2']['f']
        return rouge_2

    def get_rouge_l(self, source, target, unit='char'):
        if unit == 'word':
            source = jieba.cut(source)
            target = jieba.cut(target)
        source, target = ' '.join(source), ' '.join(target)
        score = self.rouge.get_scores(hyps=target, refs=source)
        rouge_l = score[0]['rouge-l']['f']
        return rouge_l

    def get_bleu(self, source, target, unit='char'):
        if unit == 'word':
            source = jieba.cut(source)
            target = jieba.cut(target)
        source, target = ' '.join(source), ' '.join(target)
        bleu = sentence_bleu(
            references=[source.split(' ')],
            hypothesis=target.split(' '),
            smoothing_function=self.smooth
        )
        return bleu


if __name__ == '__main__':
    a = "我去上学校,天天不迟到"
    b = "你去上学校天天都迟到"
    matrix = get_evaluation_index()
    ans = matrix.get_bleu(a, b)
    
    print(ans)