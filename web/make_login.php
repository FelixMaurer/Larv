<?php
// php web/make_login.php <user> <htpasswd_out> <plain_out>
// Random password (no ambiguous characters), bcrypt hash for Apache Basic Auth, plain text only in the
// local (git-ignored) file. Upload the hash file to /.htpasswd-larvae (webspace root, outside the webroot).
[$_, $user, $htOut, $plainOut] = $argv;
$a = 'abcdefghijkmnpqrstuvwxyz23456789';
$parts = [];
for ($p = 0; $p < 3; $p++) { $s = ''; for ($i = 0; $i < 5; $i++) $s .= $a[random_int(0, strlen($a) - 1)]; $parts[] = $s; }
$pw = implode('-', $parts);
file_put_contents($htOut, $user . ':' . password_hash($pw, PASSWORD_BCRYPT, ['cost' => 11]) . "\n");
file_put_contents($plainOut, "Larvae Explorer login (browser password prompt)\nURL:      https://felixmilanmaurer.com/larvae/\nUser:     $user\nPassword: $pw\n");
echo "written\n";
