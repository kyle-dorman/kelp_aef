from kelp_aef import main


def test_main_smoke(capsys) -> None:
    main()

    captured = capsys.readouterr()
    assert captured.out == "Hello from kelp-aef!\n"
