# import http.client
#
# conn = http.client.HTTPConnection('192.168.124.215')
#
#
# payload = {'ShipID': '28dfb30e-c555-11e7-abc4-cec278b6b50a',
#            'SystemInfo': {'SystemName': 'IRSPP01', 'Memo': 'IRSPP Pilot System'},
#            'SensorInfo': {'Level': '1', 'SmartZone': 'SZ-IRSPP', 'Hub': 'SIoT-IH-001',
#                           'HubUUID': 'b0314016-c555-11e7-abc4-cec278b6b50a',
#                           'Component': 'FIT-VRL01', 'Sensor': 'FIT-VRL01', 'UpTime': '16948'},
#            'Data': {'ID': '321880641', 'value':'689.941406'}}
#
# headers = {'content-type': 'application/json', 'cache-control': 'no-cache','postman-token': '587a4791-ea41-825a-feb9-f478788cd92f'}
#
# conn.request('POST', '/time2.php', payload, headers)
#
# res = conn.getresponse()
#
# data = res.read()
#
# print(data)

# print(data.decode('utf-8'))



