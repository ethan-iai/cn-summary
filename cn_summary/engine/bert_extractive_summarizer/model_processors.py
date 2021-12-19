# -*- coding: utf-8 -*-
from .summarizer.bert_parent import BertParent
from .summarizer.cluster_features import ClusterFeatures
from .summarizer.sentence_handler import SentenceHandler

from cn_summary.utils import get_src_path, text_legal

@text_legal
def bert_extract(text, num_sentences=None, ratio=0.2):
    summarizer = ModelProcessor()
    return summarizer(text, num_sentences, ratio)

class ModelProcessor(object):

    def __init__(self, bert_path=get_src_path('./bert-base-chinese'), min_length=20, max_length=300, pca_k=None, random_state=12345):
        """
        :param bert_path: The string path for the pretrained bert model.
        :param min_length: The minimum length a sentence should be to be considered.
        :param max_length: The maximum length a sentence should be to be considered.
        :param pca_k: If you want the features to be ran through pca, this is the components number.
        :param random_state: Random state.
        """
        self.model = BertParent(bert_path)
        self.sentence_handler = SentenceHandler(min_length, max_length)
        self.cluster_features = ClusterFeatures(pca_k, random_state)

    def cluster_runner(self, content, num_sentences, ratio):
        hidden = self.model(content)
        hidden_args = self.cluster_features(hidden, num_sentences, ratio)
        sentences = [content[j] for j in hidden_args]        
        return sentences

    def run(self, doc, num_sentences, ratio):   
        sentences = self.sentence_handler(doc)
        if sentences:
            sentences = self.cluster_runner(sentences, num_sentences, ratio)
        return '。'.join(sentences)

    def __call__(self, doc, num_sentences=None, ratio=0.2):
        return self.run(doc, num_sentences, ratio)

if __name__ == '__main__':
    
    doc = """　　从层出不穷的上市公司公告来看，苏宁的流动性危机在持续发酵。
    
    　　6月16日盘后，苏宁易购(5.590, 0.00, 0.00%)发布重大事项停牌公告，称公司实控人、控股股东张近东及股东苏宁电器正在筹划转让公司股份。为维护投资者利益，保证公平信息披露，避免股价异常波动，申请了从6月16日开市起停牌，预计停牌不超过5个交易日。停牌前，苏宁易购股价5.59元，创下8年来新低，总市值520亿元。
    
    　　屋漏偏逢连夜雨。此前一天，苏宁易购公告称，股东张近东持有公司的5.8%股份被司法冻结，牵涉5.4亿股。起因是苏宁置业集团的一笔存续信托贷款，当时融资本金不超过30亿元，苏宁电器集团、张近东及其配偶提供担保。当债权方要求苏宁置业还款，后者没还不上，张近东也就跟着受了拖累。另外，苏宁电器因部分股票质押式回购交易被平仓，被动减持1000万股，未来6个月可能减持不超过3.84亿股。
    
    　　据知情人士透露，接下来数周内可能也不会平稳，估计还会有更多类似消息被曝出。
    
    　　目前几乎所有信息都指向一点，在深圳国资和江苏国资先后确定对苏宁施以援手之后，苏宁的流动性危机仍未缓解。今年2月28日，深圳国资旗下的鲲鹏资本和深圳国际下属公司分别以96.63亿元和51.54亿元，拿下了苏宁易购15%和8%的股权，合计持股23% 。深圳国资带来的这148亿元可以解苏宁的燃眉之急，2021年内苏宁系将有160亿元债券到期，需要兑付。至6月初，江苏国资委通过设立新零售基金，“借钱”31亿元给苏宁，期限1年，明年4月还款。
    
    　　在此之前，苏宁的自救从未停止。6月15日，有员工在社交平台爆料，称苏宁易购南京总部把“员工宿舍”卖了，要求一个月内搬走。随后苏宁回复称，紫金嘉悦实际上是对外出售的商品房，并非员工宿舍，当初是把未售出的800套房子便宜租给员工，但近期随着房子售出，租户也得搬走。
    
    　　苏宁物流也被惦记上了。一家券商研究员告诉AI财经社：“去年苏宁就想把苏宁物流卖了，顺丰、京东物流、菜鸟都去看过，但苏宁想把大件物流业务留下来，只卖其他业务，潜在买家们垂涎的恰是大件物流，觉得其他业务是累赘，所以就没有进一步接触。”
    
    　　早在今年2月下旬，苏宁就把亏损严重的天天快递“临机处理”，名义上转型做同城配送，服务苏宁零售业务，实际上相当于放弃后者经营20多年的全国性网点，彻底脱离快递战场。一名苏宁物流离职仓储经理透露，以前的小件仓基本取消，只保留大件仓，物流业务收缩不少，已有很多同事离职。
    
    　　同期被舍弃的还有江苏足球俱乐部。当苏宁宣布俱乐部停止运营时，距离这支球队拿到中超冠军仅过了108天。4年前张近东斥巨资修建的徐庄训练基地，如今正在被拆除。
    
    　　某种程度上，苏宁各种变卖资产，也是为了积极还债，承担企业责任，但仓皇之下难以协调好各方利益。比如足球队停运后，俱乐部员工的欠薪和离职赔偿问题尚未得到完全解决。3月中旬，江苏队还有8名员工委托律师起诉江苏足球俱乐部，包括拖欠工资和经济补偿等，索赔305万元。
    
    　　物流业务也有遗留问题。袁华曾是苏宁物流华北区对外宣传的标杆加盟商，他透露今年1月后苏宁物流的派送费就没有结算，高达59万元，还有天天快递退网的17万元押金也没退，这对加盟商造成不小的压力。
    
    　　被苏宁收购的万达百货也出现类似的情况。北京一家苏宁易购广场（原万达百货）的耐克专卖店店员透露，商场运营方已经拖欠他们5个多月营收，高达700多万元，“听说之前欠PUMA两个月的流水，导致人家关店”，如今耐克店不再用商场配发的POS机，这样营业流水就不用通过苏宁。
    
    　　2021年上半年，苏宁一直都在缺钱中度过。2月25日，苏宁易购以公告的形式，将资金短缺的窘迫公之于众：大股东拟转让至少五分之一的股份，筹划控制权变更，“变卖家产”这一招让外界的疑惑达到了顶点，苏宁到底多缺钱？
    
    　　有统计显示，截至2020年前三季度，苏宁的流动负债总额高达1099.67亿元。据《财新周刊》报道，一位江苏当地银行高管表示，整个苏宁集团包括上市公司在内的负债可能达到2000多亿元，其中大头是银行信贷，还有一部分体量不小的公司债。
    
    　　深圳国资和江苏国资的出手，一度让苏宁看到希望。但让苏宁难受的是，钱没有那么快到手。
    
    　　“深圳国资和江苏国资的钱都还没有到位。”有接近苏宁的知情人士向AI财经社透露，“深圳国资上个月还派了一队人马到苏宁南京总部做尽职调查，目前尽调基本接近尾声，不过估计江苏国资的钱应该会更早到账，听说很快到位。到时候苏宁的现金流危机将会得到很大缓解。”该人士认为，苏宁属于短时间内的现金流紧缺，压力主要源自外部债务，内部运营尚能维持正常，员工工资照发。
    
    　　一旦两地国资的资金到位，现金流危机就能得到很大缓解。“零售企业其实不缺现金流，关键是解决外部债务。
    
    　　“不值得恐慌。”一名苏宁内部人士表示，但其并未透露江苏国资的钱何时到账。
    
    　　缺钱的日子并不好过。今年前4个月，苏宁置业已经被质押了10多次，其中银河物业是最大的一笔。4月份有消息称，苏宁将旗下江苏银河物业管理有限公司以30亿元的价格卖给了碧桂园。6月16日，苏宁易购股东苏宁电器集团有限公司向星链商业保理（天津）有限公司质押股份1.47亿股，用于为保理业务提供增信。
    
    　　据企查查APP信息显示，星链商业保理（天津）有限公司为上海苏宁金融服务集团有限公司100%控股，张近东间接以26.38%的股权占比为实际控制人。
    
    　　如此来看苏宁质押股权的公司，实质上仍是自家的资产。“严格意义上不可能出现股权质押给自己的情况，但如果是通过公司间接控制的话，一定要在法律及证监会允许下才可以。”一位不具名的金融行业人士分析道，“最坏的情况就是质押股权的公司收不回来股权，只能进行业务重组，如果苏宁本身就有业务调整的意向时，这样质押除了能够获得流动资金的支持，还能帮助企业进行调整。”
    
    　　处于十面埋伏的苏宁，也让在其中工作的员工感受到了异样气氛。“苏宁现在销售还是向好的，最怕墙倒众人推，一会儿一个公告，股价跟着跌，我们都心惊胆战的。现在的苏宁需要时间。”一位苏宁员工说。
    
    　　在深圳国资认购苏宁易购23%股权后，苏宁方面强调，苏宁创始人、董事长张近东仍是公司实际控制人。苏宁声称两家投资机构并非一致行动人，“张近东团队仍是公司的实际管理者，深圳国资方面有提名董事的权利。”苏宁高层人士曾透露道。
    
    　　在解决资金危机的同时，苏宁和张近东还要面对的一个拷问是，如何掉头。“以前烧钱太多，一口吃得太胖了，现在回血不够，难受了。”一名前苏宁易购员工有点悲观，“（公司）盘子太大，现在都得瘦身，本来就是借债滚雪球，零售、线下又是个重活儿，加上其他对手这么强，经不起折腾，填窟窿可不容易。”
    
    　　即便苏宁此次尽全力将窟窿补上，但后续业务是否能保持不再亏损，将是苏宁能否真正活下去的关键。重中之重则是苏宁的新零售业务。
    
    　　随着线下零售时代过去，苏宁也开启了转型之路。建线上商城、推母婴品牌、入局视频行业、开便利店、自营物流……苏宁一直在各个领域探索自己的转型之路。除去收购家乐福中国、万达百货，拓展电器以外的零售百货版图，苏宁的零售之路始终缺少一个“引擎”。
    
    　　零售电商行业专家、百联咨询创始人庄帅在3年前开始深入研究苏宁易购，他认为，苏宁易购在阿里入股、丧失独立性之后，便出现了四大隐忧：自营品类销售增长严重依赖在天猫开的苏宁易购店；利润来自出售阿里股份的占比过高；独立的营销、运营和技术能力已经被削弱，之前积累的人才多流失；清晰的战略和创新模式缺失。
    
    　　作为一家创立31年的零售企业，苏宁自然不会缺乏反思。今年年初，张近东重新为苏宁定调，要走轻资产路线，上一个喊出这个口号的还是他曾经大手笔驰援的万达。苏宁准备聚焦主业，深耕零售，宣布由“零售商”全面升级为“零售服务商”，此外重点发力金融、置业两个板块。今年一季度，苏宁零售云店开店600家，通过自营与加盟，线上与线下，加强零售基础能力的开放，帮助更多的中小零售商。
    
    　　除此之外，苏宁还放弃了曾经大手笔投入营销的做法。据财报显示，2020年全年，苏宁的总费用率较同期下降2.61%。其中，销售费用下降23.43%，财务费用下降12.16%，管理费用下降5.77%。
    
    　　在今年2月的新春团拜会上，苏宁集团董事长张近东表示：“2021年，苏宁将聚焦家电、自主产品、低效业务调整以及各类费用控制四个利润点，强化易购主站、零售云、B2B平台、猫宁四个规模增长源。”
    
    　　当时苏宁的流动性危机刚刚呈现出冰山一角。张近东为苏宁定下的近景目标是，要实现从商业模式向盈利模式的转变。
    
    """
    summarizer = ModelProcessor()
    print(summarizer(doc, 3))