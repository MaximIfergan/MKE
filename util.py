def print_title(title):
    res = "      " + title + "      "
    while (len(res) < 90):
        res = "=" + res + "="
    print("# " + res)


if __name__ == "__main__":
    print_title("Global Vars")
    print_title("Global functions")
