import os
import json


class Knapsack:

    def __init__(self, fpath):
        with open(fpath, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
            self.max_weight = raw_data["max_weight"]

            item_values = raw_data["item_values"]
            item_weights = raw_data["item_weights"]

            self.num_items = len(item_values)
            assert(self.num_items == len(item_weights))

            self.items = [{'value': v, 'weight': w} for v, w in zip(item_values, item_weights)]

    def evaluate_backpack(self, genome):
        """
            Determine the value of the backpack.
            Adds up the values, only returns the summed
            value when the weight is less than the threshold.
        """

        item_vector = genome.data

        # confirm that it is a valid array
        assert (len(item_vector) == self.num_items)

        total_value = 0
        total_weight = 0

        for i, has_item in enumerate(item_vector):

            # check if the value for this item is true
            if has_item:
                item = self.items[i]

                # get the corresponding value and weight for this item
                value = item['value']
                weight = item['weight']

                # add to total
                total_value += value
                total_weight += weight

        # only return the summed value if
        # the weight is not over the threshold
        # of MAX_WEIGHT
        if total_weight <= self.max_weight:
            return total_value
        else:
            return -total_weight
