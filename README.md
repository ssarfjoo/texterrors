# texterrors

For calculating WER, CER, other metrics and getting detailed statistics.  

Does character aware alignment by default, core is C++ so is fast.

Supports scoring by group (for example by speaker) or just scoring keywords or phrases and other things. Meant to replace older tools like `sclite` by being easy to use, modify and extend.

See here for [background motivation](https://ruabraun.github.io/jekyll/update/2020/11/27/On-word-error-rates.html).

Red is what the model missed (deletion or the reference word in subtitution error). Read the white and red words to read the reference.

Green is what the model output incorrectly (insertion or the hypothesis word in substitution error). Read the white and green words to read the hypothesis.

![Example](docs/images/texterrors_example.png)

# Installing
Requires minimum python 3.6! And the `pybind11` package should already be installed.
```
git clone https://github.com/ssarfjoo/texterrors.git
python -m pip install texterrors/
```
The package will be installed as `texterrors`.

# Example

The `texterrors.py` file will be in your path after running pip install.

## Command line

The `-s` option means there will be no detailed output. Below `ref` and `hyp` are files with the first field equalling the utterance ID.
```
$ texterrors.py -isark -s ref hyp
WER: 83.33 (ins 1, del 1, sub 3 / 6)
```

You can specify an output file to save the results, probably what you if you are getting detailed output.
```
$ texterrors.py -isark -cer -oov-list-f oov_list ref hyp detailed_wer_output
```
If you look at the output file with `less` use the `-R` flag to see color.

# Options you might want to use 

There are more options, call with `-h` to see.

`-utt-group-map` - Should be a file which maps uttids to group, WER will be output per group (could use
to get per speaker WER for example).

`-freq-sort` - Sort errors by frequency rather than count

`-isctm` - Will use time stamps for alignment (this will give the best one), last three columns of ctm should be time, duration, word.

`-oov-list-f` - The CER between words aligned to the OOV words will be calculated (the OOV-CER). 

`-keywords-list-f` - The reference is filtered by these keywords before calculating metrics like WER. In other words, score on just specific keyword list.

`-group-keywords-list-f` - Similar to `-keywords-list-f`, however separated per group. Mostly used when `-utt-group-map` is defined. 

`-phrase-f` - If you just want to score a phrase inside an utterance.

`-phrase-list-f` - Similar to `-phrase-f`, however score multiple phrases inside an utterance

`-class-to-word` - List of classes with words for computing the class based accuracy

# Why is the WER slightly higher than in kaldi ?

**You can make it equal by using the `-no-chardiff` argument.**

This difference is because this tool does character aware alignment. Across a normal sized test set this should result in a small difference. 

In the below example a normal WER calculation would do a one-to-one mapping and arrive at a WER of 66.67\%.

| test | sentence | okay    | words | ending | now |
|------|----------|---------|-------|--------|-----|
| test | a        | sentenc | ok    | endin  | now |

But character aware alignment would result in the following alignment:

| test | - | sentence | okay | words | ending | now |
|------|---|----------|------|-------|--------|-----|
| test | a | sentenc  | ok   | -     | endin  | now |

This results in a WER of 83.3\% because of the extra insertion and deletion. And I think one could argue this is the actually correct WER.
