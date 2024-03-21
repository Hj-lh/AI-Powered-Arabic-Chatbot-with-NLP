import sqlite3

def create_connection():
    """Create a database connection to the SQLite database."""
    conn = None
    try:
        conn = sqlite3.connect('exam_dates.db')
    except sqlite3.Error as e:
        print(e)
    return conn

def create_table(conn):
    """Create exams table if it doesn't exist."""
    create_table_sql = '''
    CREATE TABLE IF NOT EXISTS exams (
        id integer PRIMARY KEY,
        subject_code text NOT NULL,
        exam_date text NOT NULL
    );
    '''
    try:
        c = conn.cursor()
        c.execute(create_table_sql)
    except sqlite3.Error as e:
        print(e)

def insert_exam_date(conn, subject_code, exam_date):
    """Insert a new exam_date into the exams table."""
    sql = ''' INSERT INTO exams(subject_code, exam_date)
              VALUES(?, ?) '''
    cur = conn.cursor()
    cur.execute(sql, (subject_code, exam_date))
    conn.commit()
    return cur.lastrowid

def get_exam_date(conn, subject_code):
    """Query exam dates by subject_code."""
    cur = conn.cursor()
    cur.execute("SELECT exam_date FROM exams WHERE subject_code=?", (subject_code,))
    rows = cur.fetchall()
    return rows

if __name__ == '__main__':
    conn = create_connection()
    if conn is not None:
        create_table(conn)
        
        # Example: Insert exam dates. Only run once to avoid duplicates.
        insert_exam_date(conn, 'عال 111', '2024/4/23')
        insert_exam_date(conn, 'هال 333', '2024/5/10')
        
        conn.close()

