# uesf project

```bash
uesf project --help
```

```
Usage:  uesf project [OPTIONS] COMMAND

Manage UESF projects. Please ensure your terminal is in the project directory when running project commands.

Options:
  --help  Show this message and exit.
Commands:
  register    Register a project from project.yml in the current directory.
  list        List all registered projects.
  remove      Remove a registered project.
  preprocess  Run preprocessing on raw datasets defined in the project.
  train       Run model training with datasets and models defined in the project.
  evaluate    Run evaluation on trained models.
  run         Sequentially execute preprocess, train, and evaluate as needed.
  model       Manage models within the current project.
```

## uesf project register

```bash
uesf project register --help
```

```
Usage:  uesf project register [OPTIONS]
Register a new project based on project.yml in the current directory.

Options:
  --help  Show this message and exit.
```

## uesf project list

```bash
uesf project list --help
```

```
Usage:  uesf project list [OPTIONS]
List registered projects.

Options:
  --help  Show this message and exit.
  --project-name <project-name>  Specify the project name.
```

## uesf project remove

```bash
uesf project remove --help
```

```
Usage:  uesf project remove [OPTIONS] <project-name>
Remove a registered project by name.

Options:
  --help  Show this message and exit.
  --project-name <project-name>  Specify the project name.
```

## uesf project preprocess

```bash
uesf project preprocess --help
```

```
Usage:  uesf project preprocess [OPTIONS]
Run preprocessing step for datasets according to the project.yml configuration.

Options:
  --help  Show this message and exit.
```

## uesf project train

```bash
uesf project train --help
```

```
Usage:  uesf project train [OPTIONS]
Train models based on datasets and definitions in the project.yml configuration.

Options:
  --help  Show this message and exit.
```

## uesf project evaluate

```bash
uesf project evaluate --help
```

```
Usage:  uesf project evaluate [OPTIONS]
Evaluate trained models with evaluation configurations defined in the project.

Options:
  --help  Show this message and exit.
```

## uesf project run

```bash
uesf project run --help
```

```
Usage:  uesf project run [OPTIONS]
Automatically detect config modifications and sequentially run the required preprocess, train, and evaluate steps.

Options:
  --help  Show this message and exit.
```

## uesf project model

```bash
uesf project model --help
```

```
Usage:  uesf project model [OPTIONS] COMMAND

Manage models attached to the current UESF project.

Options:
  --help  Show this message and exit.

Commands:
  list        List models defined in the project.
  add         Add a new model entry point to the project locally.
  remove      Remove a model from the project locally.
```

## uesf project model list

```bash
uesf project model list --help
```

```
Usage:  uesf project model list [OPTIONS]
List models registered in the current project's configuration.

Options:
  --help  Show this message and exit.
```

## uesf project model add

```bash
uesf project model add --help
```

```
Usage:  uesf project model add [OPTIONS]
Register a new custom model to the current project's configuration (e.g. project.yml / model.yml).
The code remains in your local directory, and UESF just creates a reference to it.

Options:
  --help  Show this message and exit.
  --name <name>  Model name.
  --entrypoint <filepath:class>  Path to the model code and class name. (e.g., ./models/mymodel.py:MyModel)
```

## uesf project model remove

```bash
uesf project model remove --help
```

```
Usage:  uesf project model remove [OPTIONS] <model-name>
Remove a custom model configuration reference from your project.

Options:
  --help  Show this message and exit.
  --model-name <name>  Model name.
```
