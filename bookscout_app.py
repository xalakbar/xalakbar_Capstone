import time
import streamlit as st
from bookscout_rs import(
    insert_user,
    check_credentials, 
    get_uid,
    get_goodbooks,
    get_hy_recommendations,
)

def login():
    st.header("Log In")
    user_log = st.text_input("Username", key="login_username")
    pass_log = st.text_input("Password", key="login_password", type='password')

    if st.button("Login"):
        if user_log and pass_log:
            if check_credentials(user_log, pass_log):
                st.session_state['logged_in'] = True
                st.session_state['username'] = user_log
                st.session_state['user_id'] = get_uid(user_log)
                st.success("Login successful!")
                time.sleep(2)
                st.rerun()
            else:
                st.error("Invalid username or password.")
        else:
            st.warning("Please fill in both fields.")

def signup():
    with st.expander("Sign Up", expanded=False):
        user_sig = st.text_input("Username", key="signup_username")
        pass_sig = st.text_input("Password", key="signup_password", type="password")
        confirm_pass = st.text_input("Confirm Password", key="signup_confirm_password", type="password")

        if st.button("Sign Up"):
            if user_sig and pass_sig and confirm_pass:
                if pass_sig == confirm_pass:
                    user_id = insert_user(user_sig, pass_sig)
                    if user_id:
                        st.session_state['logged_in'] = True
                        st.session_state['username'] = user_sig
                        st.session_state['user_id'] = user_id
                        st.success("Sign-up sucessful!")
                        time.sleep(2)
                        st.rerun()
                    else:
                        st.error("Username already exists. Please choose a different one.")
                else:
                    st.error("Passwords do not match.")
            else:
                st.warning("Please fill in all fields.")

def homepage():
    goodbooks = get_goodbooks()
    titles = goodbooks['title'].tolist()

    selected_book = st.selectbox("Search or select a book", titles)

    if st.button("Get Recommendations"):
        user_id = st.session_state.get('user_id')
        work_id = goodbooks[goodbooks['title'] == selected_book]['work_id'].values[0]

        recommendations = get_hy_recommendations(user_id, work_id)

        if not recommendations.empty:
            col1, col2, col3, col4, col5 = st.columns(5)
            for index, row in recommendations.iterrows():
                col_index = index % 5
                if col_index == 0:
                    with col1:
                        st.image(row['image_url'], width=100)
                        st.text(row['title'])
                elif col_index == 1:
                    with col2:
                        st.image(row['image_url'], width=100)
                        st.text(row['title'])
                elif col_index == 2:
                    with col3:
                        st.image(row['image_url'], width=100)
                        st.text(row['title'])
                elif col_index == 3:
                    with col4:
                        st.image(row['image_url'], width=100)
                        st.text(row['title'])
                elif col_index == 4:
                    with col5:
                        st.image(row['image_url'], width=100)
                        st.text(row['title'])
            st.markdown("---")
        else:
            st.write("Sorry, no recommendations found.")


def main():
    st.title("BookScout")

    if 'logged_in' in st.session_state and st.session_state['logged_in']:
        st.sidebar.title("Welcome to BookScout!")
        st.sidebar.header(f"Hello, {st.session_state['username']}!")
        if st.sidebar.button("Logout"):
            
            for key in list(st.session_state.keys()):
                del st.session_state[key]
                st.success("Log out successful.")
                time.sleep(2)
                st.rerun()

        homepage()
    else:
        login()
        signup()

if __name__ == "__main__":
    main()