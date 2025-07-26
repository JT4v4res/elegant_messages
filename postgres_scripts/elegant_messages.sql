-- Active: 1753490194256@@127.0.0.1@5432@despedidapeu
DROP TABLE IF EXISTS elegant_messages;

CREATE TABLE elegant_messages (
    id SERIAL PRIMARY KEY,
    message_author TEXT,
    message VARCHAR,
    created_at TIMESTAMPTZ default current_timestamp
);