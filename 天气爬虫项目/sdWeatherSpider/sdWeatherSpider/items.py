import scrapy
class SdweatherspiderItem(scrapy.Item):
       # definethefieldsforyouritemherelike:
       # name=scrapy.Field()
       city=scrapy.Field()
       weather=scrapy.Field()
