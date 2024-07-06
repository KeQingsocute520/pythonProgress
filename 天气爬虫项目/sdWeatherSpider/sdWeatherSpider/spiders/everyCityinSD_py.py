from re import findall
from urllib.request import urlopen
import scrapy

from sdWeatherSpider.items import SdweatherspiderItem


class EverycityinsdSpider(scrapy.Spider):
    name = 'everyCityinSD'
    allowed_domains = ['www.weather.com.cn']
    start_urls = []
    # 遍历各城市，获取要爬取的页面
    url = r'http://www.weather.com.cn/jiangsu/index.shtml'
    with urlopen(url) as fp:
        contents = fp.read().decode()
    pattern = '<a title=".*?" href="(.+?)" target="_blank">(.+?)</a>'
    for url in findall(pattern, contents):
        start_urls.append(url[0])

    def parse(self, response, **kwargs):
        # 处理每个城市的天气预报页面数据
        item = SdweatherspiderItem()
        city = response.xpath('//div[@class="crumbs fl"]//a[3]//text()').extract()[0]
        item['city'] = city
        # 每个页面只有一个城市的天气数据，直接取[0]
        # 存放天气数据
        selector = response.xpath('//ul[@class="t clearfix"]')[0]
        weather = ''
        for li in selector.xpath('./li'):
            try:
                date = li.xpath('./h1//text()').extract()[0]
                cloud = li.xpath('./p[@title]//text()').extract()[0]
                high_tmp_list = li.xpath('./p[@class="tem"]//span//text()').extract()
                low = li.xpath('./p[@class="tem"]//i//text()').extract()[0]
                # 这里不加长度判断会导致找不到每日最高气温的时候抛出IndexError异常。
                if high_tmp_list.__len__() != 0:
                    tmp = high_tmp_list[0] + r'/' + low
                else:
                    tmp = low
                wind = li.xpath('./p[@class="win"]//em//span[1]/@title').extract()[0]
                wind = wind + li.xpath('./p[@class="win"]//i//text()').extract()[0]
                weather = weather + date + ':' + cloud + ',' + tmp + ',' + wind + '\n'
            except IndexError as e:
                continue
        item['weather'] = weather
        return [item]
