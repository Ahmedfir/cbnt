from utils.file_read_write import load_file

LINE_COMENT_START = '//'
LINE_BREAK = 'Ċ'
PARAGRAPH_COMMENT_START = '/*'
DOC_START = '/**'
DOC_END = '*/'


def cut_method(tokens, size, minimalNumberOfItemsAfterItem, item_to_keep):
    """
    /**
     * Call this to get a sublist
     * of a fixed {@code size}
     * from a given original {@code list} of items
     * including a given {@code item}
     * with - if it applies - a minimum number of items after that included item {@code minimalNumberOfItemsAfterItem}.
     * example:
     * list=["hel","lo"," ","morni","ng","!"]
     * subListIncludingItem(list," ",4,1) --> ["hel","lo"," ","morni"]
     * subListIncludingItem(list," ",4,2) --> ["lo"," ","morni","ng"]
     * subListIncludingItem(list," ",5,1) --> [hel","lo"," ","morni","ng"]
     *
     * @param tokens                          original list. It must be bigger or of the same size as {@code size}.
     * @param item_to_keep                          item to be included.
     * @param size                          size of the sublist.
     * @param minimalNumberOfItemsAfterItem minimum number of items after that included item. It must be less than size.
     * @param <T>                           Type of the items.
     * @return sublist of type {@code T} a fixed {@code size}
     * from a given original {@code list} of items
     * including a given {@code item}
     * with - if it applies - a minimum number of items after that included item {@code minimalNumberOfItemsAfterItem}.
     */
    :param tokens:
    :return:
    """
    assert tokens is not None and len(tokens) >= size, "Original list must be bigger or of the same size as size."
    assert minimalNumberOfItemsAfterItem <= size / 2, "minimalNumberOfItemsAfterItem must be less than size."
    # list already of size
    if len(tokens) == size:
        return tokens
    # // list bigger.
    listSize: int = len(tokens)
    if item_to_keep is not None:
        itemIndex: int = tokens.index(item_to_keep)
        itemsAfterCount: int = listSize - itemIndex - 1
        minimalItemsAfterCount: int = min(minimalNumberOfItemsAfterItem, itemsAfterCount)
        maximumItemsBeforeCount: int = size - minimalItemsAfterCount - 1
        startIndex: int = max(0, itemIndex - maximumItemsBeforeCount)
    else:
        startIndex: int = listSize - size
    return startIndex, tokens[startIndex: size + startIndex]


def surround_method(methodTokens, tokensBefore, tokensAfter, maximumTokensCount):
    tokensSize: int = len(methodTokens)
    missingTokens = maximumTokensCount - tokensSize
    if len(tokensBefore) < missingTokens / 2 - 1:
        result = tokensBefore + methodTokens
        if len(tokensAfter) > 0:
            tokensAfter = tokensAfter[0: min(maximumTokensCount - len(result), len(tokensAfter) - 1)]
            result += tokensAfter
    elif len(tokensAfter) < missingTokens / 2:
        maximumItemsBeforeCount = missingTokens - len(tokensAfter)
        result = []
        if len(tokensBefore) > 0:
            startIndex = max(0, len(tokensBefore) - maximumItemsBeforeCount)
            tokensBefore = tokensBefore[startIndex: len(tokensBefore) - 1]
            result += tokensBefore
        result += methodTokens
        result += tokensAfter
    else:
        if len(tokensAfter) > 0:
            maximumItemsBeforeCount = 1 + missingTokens / 2
        else:
            maximumItemsBeforeCount = missingTokens

        result = []
        if len(tokensBefore) > 0:
            startIndex = max(0, len(tokensBefore) - maximumItemsBeforeCount)
            tokensBefore = tokensBefore[int(startIndex): len(tokensBefore) - 1]
            result += tokensBefore

        result += methodTokens
        if len(tokensAfter) > 0:
            tokensAfter = tokensAfter[0: min(maximumTokensCount - len(result), len(tokensAfter) - 1)]
            result += tokensAfter
    assert len(result) <= maximumTokensCount
    assert len(result) == tokensSize + len(tokensBefore) + len(tokensAfter)

    return result, tokensBefore, tokensAfter


class FileSnippetFittingOutput:
    def __init__(self, original_len, snippet_tokens, before_tokens, after_tokens, start_cutting_index):
        self.original_len = original_len
        self.snippet_tokens = snippet_tokens
        self.before_tokens = before_tokens
        self.after_tokens = after_tokens
        self.start_cutting_index = start_cutting_index


class FileSnippet:

    def __init__(self, file_path: str, start_char: int, end_char: int):
        self.file_path = file_path
        self.start_char = start_char
        self.end_char = end_char

    def load(self, file_string=None):
        if file_string is None:
            file_string = load_file(self.file_path)
        return file_string[self.start_char: self.end_char + 1]

    def tokenize(self, cbm, file_string=None):
        return cbm.tokenize(self.load(file_string))

    def fit_max(self, cbm, max_tokens, item_to_keep, file_string=None) -> FileSnippetFittingOutput:
        if file_string is None:
            file_string = load_file(self.file_path)
        snippet_tokens = self.tokenize(cbm, file_string)
        len_tokens = len(snippet_tokens)
        start_cutting_index = -1
        before_tokens = None
        after_tokens = None
        if len_tokens < max_tokens:
            max_tokens_to_add = max_tokens - len_tokens
            method_before_str = file_string[max(0, self.start_char - max_tokens_to_add):self.start_char - 1]
            method_after_str = file_string[
                               self.end_char + 1:min(self.end_char + 1 + max_tokens_to_add, len(file_string) - 1)]
            before_tokens = [] if len(method_before_str.strip()) == 0 else cbm.tokenize(method_before_str)
            after_tokens = [] if len(method_after_str.strip()) == 0 else cbm.tokenize(method_after_str)
            snippet_tokens, before_tokens, after_tokens = surround_method(snippet_tokens, before_tokens, after_tokens,
                                                                          max_tokens)
        elif len_tokens > max_tokens:
            start_cutting_index, snippet_tokens = cut_method(snippet_tokens, max_tokens, int(max_tokens / 3),
                                                             item_to_keep=item_to_keep)
        return FileSnippetFittingOutput(len_tokens, snippet_tokens, before_tokens, after_tokens, start_cutting_index)
