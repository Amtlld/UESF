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
    description text not null
);

create table uesf.preprocessed_datasets (
    id integer primary key,
    name varchar(255) not null,
    sampling_rate integer not null,
    n_subjects integer not null,
    n_recordings integer not null,
    n_channels integer not null,
    n_samples integer not null,
    n_classes integer not null,
    electrodes text not null,
    description text not null
);

create table uesf.projects (
    id integer primary key,
    name varchar(255) not null,
    description text not null
);

create table uesf.models (
    id integer primary key,
    name varchar(255) not null,
    description text not null
);

create table uesf.evaluations (
    id integer primary key,
    name varchar(255) not null,
    description text not null
);
