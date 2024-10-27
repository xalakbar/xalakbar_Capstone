import streamlit as st
from bookscout_rs import(
    insert_user,
    check_credentials,
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
                st.success("Login successful!")
                homepage()
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
                        homepage()
                    else:
                        st.error("Username already exists. Please choose a different one.")
                else:
                    st.error("Passwords do not match.")
            else:
                st.warning("Please fill in all fields.")

def homepage(goodbooks):
    st.title("BookScout")
    st.write(f"Hello, {st.session_state.get('username', 'Guest')}!") 


   












def main():
    st.title("BookScout")

    if 'logged_in' in st.session_state and st.session_state['logged_in']:
        homepage()
    else:
        login()
        signup()

if __name__ == "__main__":
    main()