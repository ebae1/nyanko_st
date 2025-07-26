def main() -> None:
    """メイン処理"""
    df_cats = load_data(CATS_FILE, NUMERIC_COLS_CATS)
    df_cats = process_traits(df_cats)
    
    df_enemy = load_data(ENEMY_FILE, NUMERIC_COLS_ENEMY)

    selected_page = st.radio(
        label="tab",
        options=["Cats", "Enemy"],
        horizontal=True,
        label_visibility="collapsed",
    )

    if selected_page == "Cats":
        df_filtered = df_cats.copy()
        st.sidebar.title("Cats フィルター")

        if st.sidebar.checkbox('own'):
            if 'own' in df_filtered.columns:
                df_filtered = df_filtered[df_filtered['own'] > 0]

        search_text = st.sidebar.text_input("キャラクター名")
        if search_text and 'キャラクター名' in df_filtered.columns:
            df_filtered = df_filtered[df_filtered['キャラクター名'].str.contains(search_text, na=False)]

        selected_colors = st.segmented_control('対象属性', COLOR_TRAITS, selection_mode='multi')
        if selected_colors:
            mask = pd.Series(True, index=df_filtered.index)
            for color in selected_colors:
                if color in df_filtered.columns:
                    mask &= df_filtered[color]
            df_filtered = df_filtered[mask]

        selected_ranks = st.sidebar.multiselect('ランク', ['基本', 'EX', 'レア', '激レア', '超激レア', '伝説レア'])
        if selected_ranks and 'ランク' in df_filtered.columns:
            df_filtered = df_filtered[df_filtered['ランク'].isin(selected_ranks)]

        selected_ranges = st.sidebar.multiselect('単体or範囲', ['単体', '範囲'], default=['単体', '範囲'])
        if selected_ranges and '範囲' in df_filtered.columns:
            df_filtered = df_filtered[df_filtered['範囲'].isin(selected_ranges)]

        selected_effects = st.sidebar.multiselect('特殊効果',
            ['めっぽう強い', '打たれ強い', '超ダメージ', '攻撃力低下', '動きを止める',
             '動きを遅くする', 'ふっとばす', '呪い', '攻撃無効']
        )
        if selected_effects:
            mask = pd.Series(True, index=df_filtered.index)
            for effect in selected_effects:
                if effect in df_filtered.columns:
                    mask &= df_filtered[effect]
            df_filtered = df_filtered[mask]

        selected_abilities = st.sidebar.multiselect('特殊能力',
            ['波動', '小波動', '烈波', '小烈波', '爆波',
             'クリティカル', '渾身の一撃', 'ゾンビキラー', '悪魔シールド貫通', 'バリアブレイク',
             '生き残る', '波動ストッパー'])
        if selected_abilities:
            mask = pd.Series(True, index=df_filtered.index)
            for ability in selected_abilities:
                if ability in df_filtered.columns:
                    mask &= df_filtered[ability]
            df_filtered = df_filtered[mask]

        slider_columns = ['コスト', '再生産F', '速度', '射程', '発生F', '攻撃力', '頻度F', 'DPS', '体力', 'KB']
        for col in slider_columns:
            df_filtered = filter_by_slider(df_filtered, col)

        st.header("Cats DB")

        if not df_filtered.empty:
            max_vals, min_vals = safe_get_max_min(df_filtered, NUMERIC_COLS_CATS)
            columns_to_display = [col for col in DISPLAY_COLS_CATS if col in df_filtered.columns]
            display_df = df_filtered[columns_to_display]
            grid_builder = GridOptionsBuilder.from_dataframe(display_df)
            grid_builder.configure_default_column(suppressMenu=True)
            grid_builder.configure_selection(selection_mode="single")
            if 'キャラクター名' in display_df.columns:
                grid_builder.configure_column('キャラクター名', minWidth=150)
            if '特性' in display_df.columns:
                grid_builder.configure_column('特性', minWidth=300, wrapText=True, autoHeight=True)
            for col_name in ['ランク', '範囲', 'KB', 'No.', 'own', '速度']:
                if col_name in display_df.columns:
                    grid_builder.configure_column(col_name, initialWidth=100)
            grid_options = grid_builder.build()
            grid_response = AgGrid(
                display_df,
                gridOptions=grid_options,
                update_mode=GridUpdateMode.SELECTION_CHANGED,
                allow_unsafe_jscode=True,
                fit_columns_on_grid_load=True,
            )
            selected_user_rows = grid_response.get('selected_rows', [])
            
            if selected_user_rows:
                selected_series = pd.DataFrame(selected_user_rows).iloc[0]
                name = selected_series.get('キャラクター名', '')
                st.subheader(f"📊 {name} のステータス")
                draw_chart(selected_series, max_vals, min_vals, columns_to_display)
            else:
                st.info("上の表からキャラクターをクリックして選択すると、ステータスグラフが表示されます。")
        else:
            st.warning("この条件に一致するキャラクターはいません。")

    elif selected_page == "Enemy":
        df_filtered = df_enemy.copy()
        st.sidebar.title("Enemy フィルター")
        search_text = st.sidebar.text_input("敵キャラクター名")
        if search_text and 'キャラクター名' in df_filtered.columns:
            df_filtered = df_filtered[df_filtered['キャラクター名'].str.contains(search_text, na=False)]
        st.header("Enemy DB")
        if not df_filtered.empty:
            max_vals, min_vals = safe_get_max_min(df_filtered, NUMERIC_COLS_ENEMY)
            columns_to_display = [col for col in df_filtered.columns]
            grid_builder = GridOptionsBuilder.from_dataframe(df_filtered)
            grid_builder.configure_default_column(suppressMenu=True, filter=False)
            grid_builder.configure_selection(selection_mode="single")
            grid_options = grid_builder.build()
            grid_response = AgGrid(
                df_filtered,
                gridOptions=grid_options,
                update_mode=GridUpdateMode.SELECTION_CHANGED,
                allow_unsafe_jscode=True,
                fit_columns_on_grid_load=True,
            )
            selected_user_rows = grid_response.get('selected_rows', [])
            
            if selected_user_rows:
                selected_series = pd.DataFrame(selected_user_rows).iloc[0]
                name = selected_series.get('キャラクター名', '')
                st.subheader(f"📊 {name} のステータス")
                draw_chart(selected_series, max_vals, min_vals, NUMERIC_COLS_ENEMY)
        else:
            st.warning("この条件に一致する敵キャラクターはいません。")

if __name__ == "__main__":
    main()