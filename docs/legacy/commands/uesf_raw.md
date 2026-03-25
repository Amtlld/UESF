# uesf raw

```bash
uesf raw --help
```

```
Usage:  uesf raw [OPTIONS] COMMAND

Options:
  --help  Show this message and exit.
Commands:
  import  Import a raw dataset.
  list  List raw dataset(s).
  remove  Remove raw dataset(s).
  preprocess  Preprocess a raw dataset.
```

## uesf raw import

```bash
uesf raw import --help
```

```
Usage:  uesf raw import [OPTIONS]

Options:
  --help  Show this message and exit.
  --raw-path <path>  Path to the raw dataset.

Dataset Info (Recommended to provide in raw.yml):
  --raw-name <name>  Name of the raw dataset.
  --raw-sampling-rate <sampling rate> Sampling rate of the raw dataset.
  --raw-n-subjects <number of subjects>
  --raw-n-recordings <number of recordings>
  --raw-n-channels <number of channels>
  --raw-n-samples <number of samples>
  --raw-n-classes <number of classes>
  --raw-electrodes <list of electrodes> List of electrodes. Length must be equal to raw-n-channels.
  --raw-description <description> Description of the raw dataset.
```

## uesf raw list

```bash
uesf raw list --help
```

```
Usage:  uesf raw list [OPTIONS]
List raw dataset(s) with given conditions.

Options:
  --help  Show this message and exit.
  --raw-name <name>  Name of the raw dataset.
  --raw-sampling-rate <sampling rate> Sampling rate of the raw dataset.
  --raw-n-subjects <number of subjects>
  --raw-n-recordings <number of recordings>
  --raw-n-channels <number of channels>
  --raw-n-samples <number of samples>
  --raw-n-classes <number of classes>
  --raw-electrodes <list of electrodes> List of electrodes. Length must be equal to raw-n-channels.
```

## uesf raw remove

```bash
uesf raw remove --help
```

```
Usage:  uesf raw remove [OPTIONS]
Remove raw dataset(s) with given conditions.

Options:
  --help  Show this message and exit.
  --raw-name <name>  Name of the raw dataset.
  --raw-id <id>  ID of the raw dataset.
```

## uesf raw preprocess

```bash
uesf raw preprocess --help
```

```
Usage:  uesf raw preprocess [OPTIONS] <raw-dataset-name>
Preprocess a raw dataset.

Options:
  --help  Show this message and exit.
  --config <path>  Path to the preprocess configuration file.
  --output-name <name>  Name of the output preprocessed dataset.
```