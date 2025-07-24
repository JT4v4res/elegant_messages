DROP TABLE IF EXISTS chat_history;

CREATE TABLE chat_history (
    id SERIAL PRIMARY KEY,
    session_id uuid,
    message JSONB,
    created_at TIMESTAMPTZ default current_timestamp
);