import pytest

from second_module_home_work_with_decorator import MyCountVectorizer

from tests.test_mycountvectorizer_data import (
    corpus1, matrix1, names1,
    corpus2, matrix2, names2,
)


@pytest.fixture
def count_vectorizer():
    c = MyCountVectorizer()
    yield c


def test_empty_corpus(count_vectorizer: MyCountVectorizer):
    res = count_vectorizer.fit_transform([])
    assert res == []
    assert count_vectorizer.get_feature_names() == []


@pytest.mark.parametrize('corpus, matrix, names',
                         [(corpus1, matrix1, names1),
                          (corpus2, matrix2, names2)])
def test_normal_corpus(
        count_vectorizer: MyCountVectorizer,
        corpus: list[str],
        matrix: list[list[int]],
        names: list[str],
):
    res = count_vectorizer.fit_transform(corpus)
    assert res == matrix
    assert count_vectorizer.get_feature_names() == names


def test_one_suggestion(count_vectorizer: MyCountVectorizer):
    res = count_vectorizer.fit_transform(['one two three four'])
    assert res == [[1, 1, 1, 1]]
    assert count_vectorizer.get_feature_names() == ['four', 'one', 'three', 'two']


def test_one_suggestion_several_entry(count_vectorizer: MyCountVectorizer):
    res = count_vectorizer.fit_transform(['one two one three two one'])
    assert res == [[3, 1, 2]]
    assert count_vectorizer.get_feature_names() == ['one', 'three', 'two']
