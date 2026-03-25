create schema uesf;

create table uesf.raw_datasets (
    id integer primary key,
    name varchar(255) not null,
    sampling_rate integer not null,
    n_subjects integer not null,
    n_recordings integer not null,
    n_channels integer not null,
    n_samples integer not null,
    n_classes integer not null,
    electrodes text not null,
    description text,
    path text not null
);

create table uesf.preprocessed_datasets (
    id integer primary key,
    raw_id integer references uesf.raw_datasets(id),
    name varchar(255) not null,
    preprocess_config_path text not null,
    preprocess_config_content text not null,
    sampling_rate integer not null,
    n_subjects integer not null,
    n_recordings integer not null,
    n_channels integer not null,
    n_samples integer not null,
    n_classes integer not null,
    electrodes text not null,
    description text not null
);

-- create table uesf.projects (
--     id integer primary key,
--     name varchar(255) not null,
--     description text not null
-- );
