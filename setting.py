import json
import os
import sys


class FileController:
    def __init__(self) -> None:
        self.path = self.resource_path("setting.json")
        self.init_json()

    def init_json(self):
        json_string = {
            "REPLICATE_API_TOKEN": "",
        }

        if not os.path.isfile(self.path):
            with open(self.path, "w", encoding="UTF-8-sig") as outfile:
                json.dump(json_string, outfile, indent=4, ensure_ascii=False)

    def resource_path(self, relative_path):
        base_path = getattr(sys, "_MEIPASS", os.path.dirname(os.path.abspath(__file__)))
        return os.path.join(base_path, relative_path)

    def revise_str_json(self, key, value):
        data = self.load_json()
        data[key] = value
        with open(self.path, "w", encoding="UTF-8-sig") as outfile:
            json.dump(data, outfile, indent=4, ensure_ascii=False)

    def add_dict_json(self, key, value_key, val):
        data = self.load_json()
        data[key][value_key] = val
        with open(self.path, "w", encoding="UTF-8-sig") as outfile:
            json.dump(data, outfile, indent=4, ensure_ascii=False)

    def remove_dict_json(self, current_name):
        data = self.load_json()
        del data["camera_list"][current_name]
        with open(self.path, "w", encoding="UTF-8-sig") as outfile:
            json.dump(data, outfile, indent=4, ensure_ascii=False)

    def revise_key_dict_json(self, dict_key, current_name, new_name):
        data = self.load_json()
        data[dict_key][new_name] = data[dict_key].pop(current_name)
        with open(self.path, "w", encoding="UTF-8-sig") as outfile:
            json.dump(data, outfile, indent=4, ensure_ascii=False)

    def revise_val_dict_json(self, dict_key, val_key, value):
        data = self.load_json()
        data[dict_key][val_key] = value
        with open(self.path, "w", encoding="UTF-8-sig") as outfile:
            json.dump(data, outfile, indent=4, ensure_ascii=False)

    def load_json(self):
        data = None
        with open(self.path, "r", encoding="UTF-8-sig") as f:
            data = json.load(f)
        return data
