'''
Python Homework with Chipotle data
https://github.com/TheUpshot/chipotle
'''

'''
BASIC LEVEL
PART 1: Read in the file with csv.reader() and store it in an object called 'file_nested_list'.
Hint: This is a TSV file, and csv.reader() needs to be told how to handle it.
      https://docs.python.org/2/library/csv.html
'''

import csv
# specify that the delimiter is a tab character
# make 'file_nested_list' = list of rows

with open( 'data/order.tsv', mode = 'rU') as f:
    file_nested_list = [row for row in csv.reader( f, delimiter='\t')]

'''
BASIC LEVEL
PART 2: Separate 'file_nested_list' into the 'header' and the 'data'.
'''
# Index references for order structure
idx_order_id = 0
idx_order_qty = 1
idx_order_item = 2
idx_order_desc = 3
idx_order_price = 4

# separate the header and data
header = file_nested_list[0]
data = file_nested_list[1:]

print "Header:".rjust(12), header
print "Row Index:".rjust(12), [idx_order_id, idx_order_qty, idx_order_item, idx_order_desc, idx_order_price]
print "Sample Data:"
for row in data[:5]: print row

# Minor cleansing of data - Appears to be some significance to '[..]' - but not for us...currently
import re

for row in data:
    row[idx_order_desc] = re.sub(r'[\[\]]', '', row[idx_order_desc])
    if row[idx_order_desc] == 'NULL': row[idx_order_desc] = ''

print "Post Cleanse:"
for row in data[:10]: print row


'''
INTERMEDIATE LEVEL
PART 3: Calculate the average price of an order.
Hint: Examine the data to see if the 'quantity' column is relevant to this calculation.
Hint: Think carefully about the simplest way to do this!

order_id	quantity	item_name	choice_description	item_price
0        1       2            3                    4

'''
# count the number of unique order_id's
# note: you could assume this is 1834 since that's the maximum order_id, but it's best to check
order_ids = set([row[idx_order_id] for row in data])
print "No. Unique Orders:".rjust(30), str(len(order_ids)).rjust(10)

# create a list of prices
# note: ignore the 'quantity' column because the 'item_price' takes quantity into account
# strip the dollar sign and trailing space
order_item_price = [float(re.sub(r'[^\d.]', '', row[idx_order_price])) for row in data]

#print order_item_price
orders_total_price = sum(order_item_price)
avg_order_price = round(orders_total_price / len(order_ids),2)

print "Orders totals Value $".rjust(30), str(orders_total_price).rjust(10)
print "Average Order $".rjust(30), str(avg_order_price).rjust(10)

# calculate the average price of an order and round to 2 digits
# $18.81


'''
INTERMEDIATE LEVEL
PART 4: Create a list (or set) of all unique sodas and soft drinks that they sell.
Note: Just look for 'Canned Soda' and 'Canned Soft Drink', and ignore other drinks like 'Izze'.
'''

# if 'item_name' includes 'Canned', append 'choice_description' to 'sodas' list
sodas = []
for row in data:
    if 'canned' in row[idx_order_item].lower() and row[idx_order_desc].lower() not in sodas:
        sodas.append(row[idx_order_desc].lower())

print "For/If:".rjust(20), sorted(sodas)

# equivalent list comprehension (using an 'if' condition)
sodas = set()
sodas = set([row[idx_order_desc].lower() for row in data if 'canned' in row[idx_order_item].lower()])

print "List Comprehension:".rjust(20), sorted(sodas)

# create a set of unique sodas
sodas = set()
for row in data:
    if 'canned' in row[idx_order_item].lower():
        sodas.add( row[idx_order_desc].lower())

print "For/Set.Add:".rjust(20), sorted(sodas)



'''
ADVANCED LEVEL
PART 5: Calculate the average number of toppings per burrito.
Note: Let's ignore the 'quantity' column to simplify this task.
Hint: Think carefully about the easiest way to count the number of toppings!
'''
burrito_orders = [row for row in data if 'burrito' in row[idx_order_item].lower()]

burrito_all_toppings = []
for row in burrito_orders:
    burrito_all_toppings += [item.strip() for item in row[idx_order_desc].split(',')]

# keep a running total of burritos and toppings
print "Burrito Orders:"
for row in burrito_orders[:5]: print "  ", row

print "All Toppings:"
for row in burrito_all_toppings[:5]: print "  ", row

# calculate number of toppings by counting the commas and adding 1
# note: x += 1 is equivalent to x = x + 1
print "Total Burrito Orders:".rjust(30),  str( len( burrito_orders)).rjust(10)
print "Total Toppings Served:".rjust(30), str( len( burrito_all_toppings)).rjust(10)

# calculate the average topping count and round to 2 digits
# 5.40
print "Average Toppings Per Burrito:".rjust(30), str(round(float(len(burrito_all_toppings)) / len(burrito_orders), 3)).rjust(10)
print "No. of Burrito Types:".rjust(30),str(len(set([row[idx_order_item].lower() for row in burrito_orders]))).rjust(10)
print "Topping Types:".rjust(30), str(len(set([topping.lower() for topping in burrito_all_toppings]))).rjust(10)

'''
ADVANCED LEVEL
PART 6: Create a dictionary in which the keys represent chip orders and
  the values represent the total number of orders.
Expected output: {'Chips and Roasted Chili-Corn Salsa': 18, ... }
Note: Please take the 'quantity' column into account!
Optional: Learn how to use 'defaultdict' to simplify your code.
'''

# start with an empty dictionary

# if chip order is not in dictionary, then add a new key/value pair
# if chip order is already in dictionary, then update the value for that key
# defaultdict saves you the trouble of checking whether a key already exists
from collections import defaultdict

chips = defaultdict(int)

for row in data:
    if 'chips' in row[idx_order_item].lower():
        chips[row[idx_order_item].lower()] += int(row[idx_order_qty])

for key, value in sorted(chips.iteritems()):
    print key.rjust(48), ":", str(value).rjust(4)


'''
BONUS: Think of a question about this data that interests you, and then answer it!
'''
print "Fix the Qty Issue...."
burrito_orders = [row for row in data if 'burrito' in row[idx_order_item].lower()]
burrito_orders_flat = []
track_count = 0
for order in burrito_orders:
    track_count += int(order[1])
    for qty in range(int(order[1])):
        order[1]='1'
        burrito_orders_flat.append(order)

burrito_all_toppings=[]
for row in burrito_orders_flat: burrito_all_toppings += [item.strip() for item in row[idx_order_desc].split(',')]

# keep a running total of burritos and toppings
print "Burrito Orders:"
for row in burrito_orders_flat[:25]: print "  ",row
print "All Toppings:"
for row in burrito_all_toppings[:25]: print "  ",row

# calculate number of toppings by counting the commas and adding 1
# note: x += 1 is equivalent to x = x + 1
print "Total Burrito Orders:".rjust(30), str(len(burrito_orders_flat)).rjust(10),"( vs",track_count,")"
print "Total Toppings Served:".rjust(30), str(len(burrito_all_toppings)).rjust(10)
# calculate the average topping count and round to 2 digits
# 5.40
print "Average Toppings Per Burrito:".rjust(30),str(round(float(len(burrito_all_toppings))/len(burrito_orders_flat),3)).rjust(10)

burrito_types = set([row[idx_order_item].lower() for row in burrito_orders_flat])
burrito_ordered = defaultdict(int)
for order in burrito_orders_flat:
    burrito_ordered[order[2].lower()] += 1

print "No. of Burrito Types:".rjust(30),str(len(burrito_types)).rjust(10)
print "Qty".rjust(15),"Burrito"
checksum = 0
for burrito in sorted(burrito_types):
    print str(burrito_ordered[burrito]).rjust(15), burrito
    checksum += burrito_ordered[burrito]
print "Burrito Orders Checksum:",checksum

topping_types = set([topping.lower() for topping in burrito_all_toppings])
topping_ordered = defaultdict(int)
for topping in burrito_all_toppings:
    topping_ordered[topping.lower()] += 1

print "Topping Types:".rjust(30),str(len(topping_types)).rjust(10)
print "Qty".rjust(15),"Topping"
checksum = 0
for topping in sorted(topping_types):
    print str(topping_ordered[topping]).rjust(15), topping
    checksum += topping_ordered[topping]
print "Topping Orders Checksum:",checksum

print '====================================='
print 'playing with collection / counter'

from collections import Counter
chips = Counter([row[idx_order_item] for row in data if "Chips" in row[idx_order_item]])
for key, value in sorted(chips.iteritems()):
    print key.rjust(48),":",str(value).rjust(4)
    
    
