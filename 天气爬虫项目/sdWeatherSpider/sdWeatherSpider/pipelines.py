class SdweatherspiderPipeline:
    def process_item(self, item, spider):
      # 每一个段落执行完毕后都会调用此函数，所以文件写入模式为`a`，追加模式
        with open('weather.txt', 'a', encoding='utf8') as fp:
            fp.write(item['city'] + '\n')
            fp.write(item['weather'] + '\n\n')
        return item
