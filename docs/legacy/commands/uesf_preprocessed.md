# uesf preprocessed

```bash
uesf preprocessed --help
```

```
Usage:  uesf preprocessed [OPTIONS] COMMAND

Options:
  --help  Show this message and exit.
Commands:
  list  List preprocessed dataset(s).
  remove  Remove preprocessed dataset(s).
  add_description  Add description to a preprocessed dataset.
```

## uesf preprocessed list

```bash
uesf preprocessed list --help
```

```
Usage:  uesf preprocessed list [OPTIONS]
List preprocessed dataset(s) with given conditions.

Options:
  --help  Show this message and exit.
  --preprocessed-name <name>  Name of the preprocessed dataset.
  --preprocessed-id <id>  ID of the preprocessed dataset.
  --preprocessed-sampling-rate <sampling rate> Sampling rate of the preprocessed dataset.
  --preprocessed-n-subjects <number of subjects> Number of subjects in the preprocessed dataset.
  --preprocessed-n-recordings <number of recordings> Number of recordings in the preprocessed dataset.
  --preprocessed-n-channels <number of channels> Number of channels in the preprocessed dataset.
  --preprocessed-n-samples <number of samples> Number of samples in the preprocessed dataset.
  --preprocessed-n-classes <number of classes> Number of classes in the preprocessed dataset.
  --preprocessed-electrodes <list of electrodes> List of electrodes. Length must be equal to preprocessed-n-channels.
  --preprocessed-description <description> Description of the preprocessed dataset.
```

## uesf preprocessed remove

```bash
uesf preprocessed remove --help
```

```
Usage:  uesf preprocessed remove [OPTIONS]
Remove preprocessed dataset(s) with given conditions.

Options:
  --help  Show this message and exit.
  --preprocessed-name <name>  Name of the preprocessed dataset.
  --preprocessed-id <id>  ID of the preprocessed dataset.
```

## uesf preprocessed add_description

```bash
uesf preprocessed add_description --help
```

```
Usage:  uesf preprocessed add_description [OPTIONS]
Add description to a preprocessed dataset.

Options:
  --help  Show this message and exit.
  --preprocessed-name <name>  Name of the preprocessed dataset.
  --preprocessed-id <id>  ID of the preprocessed dataset.
  --preprocessed-description <description> Description of the preprocessed dataset.
```
